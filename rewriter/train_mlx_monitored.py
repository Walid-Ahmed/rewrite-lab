"""MLX training with local loss history, curves, and optional W&B reporting."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace


def plot_history(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, label in [('train_loss', 'Training loss (report-window average)'),
                          ('val_loss', 'Validation loss')]:
        points = [(r['iteration'], r[metric]) for r in rows if metric in r]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker='o', markersize=3, label=label)
    ax.set(xlabel='Microbatch iteration', ylabel='Cross-entropy loss', title='Qwen3 1.7B training')
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path.parent / 'loss-curves.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).resolve().parent/'mlx-qwen3-active.yaml'))
    p.add_argument('--data', type=Path, help='Explicit managed Model 1 export; otherwise use active registry')
    p.add_argument('--dataset-registry', type=Path)
    p.add_argument('--check-data-only', action='store_true', help='Validate data/provenance and exit without loading weights or starting W&B')
    p.add_argument('--iters', type=int)
    p.add_argument('--adapter-path')
    p.add_argument('--resume-adapter-file')
    p.add_argument('--offline', action='store_true', help='Record W&B locally without account login')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--ai-eval', action='store_true', help='Generate validation rewrites and a final before/after detector report')
    p.add_argument('--ai-detector-command', help='JSON argv command for detector scoring; omit to export samples for browser scoring')
    p.add_argument('--ai-detector-id')
    p.add_argument('--ai-detector-scores', help='Exact-text score file recorded through a signed-in detector browser session')
    p.add_argument('--ai-eval-every', type=int, default=0, help='Evaluate a fixed subset at validation callbacks every N iterations; 0 means final only')
    p.add_argument('--ai-eval-samples', type=int, default=8)
    p.add_argument('--ai-eval-max-tokens', type=int, default=768)
    options = p.parse_args()
    if options.ai_detector_command and not options.ai_detector_id:
        p.error('Set both --ai-detector-command and --ai-detector-id')
    if options.ai_detector_command and options.ai_detector_scores:
        p.error('Select a detector command or score file')
    if options.ai_eval_every < 0 or options.ai_eval_samples < 1 or options.ai_eval_max_tokens < 1:
        p.error('Evaluation interval must be nonnegative; sample and token counts must be positive')
    options.ai_eval = options.ai_eval or bool(options.ai_detector_command) or bool(options.ai_detector_scores) or options.ai_eval_every > 0
    import yaml
    from training_data import load_training_data, save_dataset_provenance
    config = yaml.safe_load(Path(options.config).read_text()) or {}
    selected = options.data or config.get('data')
    bundle = load_training_data('model1', selected, options.dataset_registry)
    config['data'] = str(bundle.path)
    config['dataset_provenance'] = bundle.metadata
    if options.check_data_only:
        print(json.dumps(bundle.metadata, indent=2))
        return
    import math
    import numpy as np
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.utils import load
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.tuner.callbacks import TrainingCallback
    config = {**CONFIG_DEFAULTS, **config}
    if config.get('iters') is None:
        config['iters'] = math.ceil(len(bundle.splits['train']) / config['batch_size'])
    for name in ['iters', 'adapter_path', 'resume_adapter_file']:
        value = getattr(options, name)
        if value is not None:
            config[name] = value
    config['ai_evaluation'] = {'enabled': options.ai_eval, 'every': options.ai_eval_every,
                             'periodic_samples': options.ai_eval_samples, 'max_tokens': options.ai_eval_max_tokens,
                             'detector_id': options.ai_detector_id, 'score_file': options.ai_detector_scores}
    out = Path(config['adapter_path'])
    out.mkdir(parents=True, exist_ok=True)
    history = out / 'loss-history.jsonl'
    # A continuation needs its own directory to keep curves/checkpoints unambiguous.
    if history.exists() or (out / 'adapters.safetensors').exists():
        raise ValueError('Choose a new --adapter-path for each run; use --resume-adapter-file to continue.')
    save_dataset_provenance(out, bundle.metadata)
    if bundle.metadata.get('system_prompt'):
        (out/'system-prompt.txt').write_text(bundle.metadata['system_prompt']+'\n')
    (out/'run-config.json').write_text(json.dumps(config,indent=2)+'\n')
    run = None
    if not options.no_wandb:
        import wandb
        run = wandb.init(project=config.get('project_name') or 'qwen3-humanizer',
                         name=out.name, dir=str(out.resolve()), config=config,
                         mode='offline' if options.offline else 'online')
        run.define_metric('iteration')
        run.define_metric('*', step_metric='iteration')
        (out / 'wandb-run.json').write_text(json.dumps({'id': run.id, 'entity': run.entity,
                                                     'project': run.project, 'url': run.url}, indent=2)+'\n')
        print(f'W&B run: {run.url or "offline"}')
    started = time.perf_counter()
    ai_evaluator = None

    class Callback(TrainingCallback):
        def record(self, metrics):
            serial = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in metrics.items()}
            serial['elapsed_seconds'] = time.perf_counter() - started
            with history.open('a') as f:
                f.write(json.dumps(serial) + '\n')
            if run:
                run.log(serial)
        def on_train_loss_report(self, train_info):
            self.record(train_info)
        def on_val_loss_report(self, val_info):
            # MLX reports validation before that iteration's update, using it - 1.
            self.record(val_info)
            iteration = val_info['iteration']
            # Called before update: N completed iterations is iteration >= N,
            # not necessarily divisible by N under MLX's pre-update callback.
            if ai_evaluator and options.ai_eval_every and (iteration == 0 or iteration >= self.next_ai_iteration):
                ai_evaluator.evaluate(iteration, options.ai_eval_samples, 'periodic', run)
                self.next_ai_iteration = iteration + options.ai_eval_every

        next_ai_iteration = options.ai_eval_every

    np.random.seed(config['seed'])
    args = SimpleNamespace(**config)
    try:
        model, tokenizer = load(args.model)
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        if options.ai_eval:
            from ai_content_eval import CommandDetector, FileDetector, ValidationAIEvaluator
            detector = CommandDetector(json.loads(options.ai_detector_command), options.ai_detector_id,
                                       out / 'ai-evaluation/detector-cache.json') if options.ai_detector_command else (FileDetector(options.ai_detector_scores,options.ai_detector_id) if options.ai_detector_scores else None)
            ai_evaluator = ValidationAIEvaluator(model, tokenizer, args.data, out / 'ai-evaluation', detector,
                                                options.ai_eval_max_tokens)
        callback = Callback()
        train_model(args, model, train_set, valid_set, callback)
        # MLX's built-in last validation runs before the last update. Evaluate
        # the actual saved final weights after every update has completed.
        from mlx_lm.tuner.trainer import evaluate
        from mlx_lm.tuner.datasets import CacheDataset
        final_val_loss = evaluate(model, CacheDataset(valid_set), args.batch_size,
                                  -1, max_seq_length=args.max_seq_length)
        callback.record({'iteration': args.iters, 'val_loss': final_val_loss,
                         'final_validation': True})
        summary = {'training_examples': len(train_set), 'validation_examples': len(valid_set),
                   'iterations': args.iters, 'epochs': args.iters * args.batch_size / len(train_set),
                   'fresh_adapter': args.resume_adapter_file is None,
                   'final_validation_loss': final_val_loss,
                   'elapsed_seconds': time.perf_counter() - started,
                   'dataset_version': bundle.metadata['dataset_version'],
                   'dataset_fingerprint': bundle.metadata['dataset_fingerprint'],
                   'test_examples_reserved': len(bundle.splits['test'])}
        (out / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if run:
            run.summary.update(summary)
        print('Training summary:', json.dumps(summary))
        if ai_evaluator:
            report = ai_evaluator.evaluate(args.iters, scope='final', run=run)
            print('Final AI-content evaluation:', json.dumps(report['summary']))
    finally:
        try:
            if history.exists():
                plot_history(history)
                print(f'Loss curves: {out / "loss-curves.png"}')
        finally:
            if run:
                run.finish()


if __name__ == '__main__':
    main()
