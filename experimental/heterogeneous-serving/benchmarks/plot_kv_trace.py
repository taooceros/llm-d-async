"""Per-step KV occupancy and preemption behaviour across scheduling policies.

Re-analyses recorded host block-pool traces; generating figures adds no serving work.
Collecting those traces does have overhead, qualified by paired hardware timing.
These are allocation counters, not physical HBM telemetry.

Emits, per arm: KV occupancy against wall-clock with preemption events marked, scheduled
batch, and the queue decomposition (running / waiting / undispatched).
"""
import argparse
import glob
import json
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_runs(paths):
    runs = []
    for path in sorted(paths):
        with open(path) as handle:
            document = json.load(handle)
        measurements = document.get('measurements')
        if not isinstance(measurements, dict):
            continue
        for case in measurements.get('policy_cases') or []:
            for run in case.get('runs') or []:
                if run.get('iterations'):
                    runs.append(run)
    return runs


def series(run):
    rows = run['iterations']
    total = rows[0]['total_kv_blocks'] - 1  # Exclude the permanently reserved null block.
    time = [row['scheduled_at_s'] for row in rows]
    occupancy = [total - row['free_kv_blocks_after_schedule'] for row in rows]
    batch = [row['scheduled_requests'] for row in rows]
    running = [row['running_requests'] for row in rows]
    waiting = [row['waiting_requests'] for row in rows]
    undispatched = [row['undispatched_requests'] for row in rows]
    preempt_t = [row['scheduled_at_s'] for row in rows if row['preempted_request_ids']]
    threshold = statistics.median(b for b in batch if b > 0) / 2
    drain_start = next(time[i] for i in range(len(batch) - 1, -1, -1) if batch[i] >= threshold)
    preempt_y = [total - row['free_kv_blocks_after_schedule']
                 for row in rows if row['preempted_request_ids']]
    return dict(total=total, t=time, occ=occupancy, batch=batch, running=running,
                waiting=waiting, undispatched=undispatched, drain_start=drain_start,
                drain_s=run['wall_s'] - drain_start,
                preempt_t=preempt_t, preempt_y=preempt_y)


def plot(runs, out_path, title):
    runs = sorted(runs, key=lambda r: r['wall_s'])
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 2.6 * len(runs)), sharex=True)
    if len(runs) == 1:
        axes = [axes]
    for axis, run in zip(axes, runs):
        data = series(run)
        name = run['policy']['name']
        axis.fill_between(data['t'], data['occ'], step='post', alpha=0.30,
                          color='tab:blue', linewidth=0)
        axis.plot(data['t'], data['occ'], color='tab:blue', linewidth=0.5)
        axis.axhline(data['total'], color='k', linestyle='--', linewidth=0.8)
        axis.axvspan(data['drain_start'], run['wall_s'], color='tab:purple', alpha=0.12)
        if data['preempt_t']:
            axis.scatter(data['preempt_t'], data['preempt_y'], s=14, color='tab:red',
                         zorder=5, label=f"preemption ({len(data['preempt_t'])} steps)")
            axis.legend(loc='lower left', fontsize=7, framealpha=0.9)
        axis.set_ylim(0, data['total'] * 1.06)
        axis.set_ylabel('allocated KV (null excluded)', fontsize=8)
        twin = axis.twinx()
        twin.plot(data['t'], data['batch'], color='tab:orange', linewidth=0.6, alpha=0.85)
        twin.set_ylabel('batch', color='tab:orange', fontsize=8)
        twin.tick_params(axis='y', labelcolor='tab:orange', labelsize=7)
        occupied = sum(data['occ']) / len(data['occ'])
        axis.set_title(
            f"{name}  [track {run['track']}]   wall {run['wall_s']:.1f}s   "
            f"mean batch {run['mean_scheduled_requests']:.1f}   "
            f"preemptions {run['preemption_events']}   "
            f"mean KV {occupied:.0f}/{data['total']} ({100 * occupied / data['total']:.0f}%)  "
            f"drain {data['drain_s']:.1f}s ({100 * data['drain_s'] / run['wall_s']:.1f}%)  "
            f"repeat {run.get('repeat', 0)}",
            fontsize=9, loc='left')
        axis.tick_params(labelsize=7)
    axes[-1].set_xlabel('wall-clock seconds from dispatch start', fontsize=8)
    fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def plot_queue(runs, out_path, title):
    runs = sorted(runs, key=lambda r: r['wall_s'])
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 2.2 * len(runs)), sharex=True)
    if len(runs) == 1:
        axes = [axes]
    for axis, run in zip(axes, runs):
        data = series(run)
        axis.stackplot(data['t'], data['running'], data['waiting'], data['undispatched'],
                       labels=['running (engine)', 'waiting (engine)', 'undispatched (dispatcher)'],
                       colors=['tab:green', 'tab:orange', 'tab:grey'], alpha=0.85)
        axis.set_ylabel('requests', fontsize=8)
        axis.set_title(f"{run['policy']['name']}  wall {run['wall_s']:.1f}s", fontsize=9, loc='left')
        axis.tick_params(labelsize=7)
    axes[0].legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('wall-clock seconds from dispatch start', fontsize=8)
    fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def plot_drain(runs, out_path, title):
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 2.4 * len(runs)))
    if len(runs) == 1:
        axes = [axes]
    for axis, run in zip(axes, sorted(runs, key=lambda r: r['wall_s'])):
        data = series(run)
        selected = [i for i, t in enumerate(data['t']) if t >= data['drain_start']]
        times = [data['t'][i] - data['drain_start'] for i in selected]
        axis.step(times, [data['batch'][i] for i in selected], where='post', color='tab:orange')
        axis.set_ylabel('scheduled requests')
        axis.set_xlim(0, max(data['drain_s'], 0.001))
        twin = axis.twinx()
        twin.step(times, [data['occ'][i] for i in selected], where='post', color='tab:blue', alpha=0.6)
        twin.set_ylabel('allocated KV blocks')
        twin.set_ylim(0, data['total'])
        axis.set_title(f"{run['policy']['name']} repeat {run.get('repeat', 0)}: "
                       f"drain {data['drain_s']:.3f}s / wall {run['wall_s']:.3f}s", fontsize=9)
        axis.set_xlabel('seconds from last step at least half median executed batch')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tag', default='run')
    parser.add_argument('--title', default='KV occupancy and preemption behaviour')
    args = parser.parse_args()
    runs = load_runs(glob.glob(args.glob))
    if not runs:
        raise SystemExit(f'no runs with per-step iterations matched {args.glob}')
    os.makedirs(args.out_dir, exist_ok=True)
    plot(runs, os.path.join(args.out_dir, f'{args.tag}_kv_occupancy.png'), args.title)
    plot_queue(runs, os.path.join(args.out_dir, f'{args.tag}_queue.png'),
               args.title + ' - queue decomposition')
    plot_drain(runs, os.path.join(args.out_dir, f'{args.tag}_drain.png'), args.title + ' - drain')


if __name__ == '__main__':
    main()
