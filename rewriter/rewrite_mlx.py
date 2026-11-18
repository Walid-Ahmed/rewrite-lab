"""Use the locally trained MLX Qwen3 adapter."""
import argparse
from humanizer_prompts import MODES, VIEWS, inference_system_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paragraph')
    parser.add_argument('--adapter', default='mlx-qwen3-humanizer')
    parser.add_argument('--max-tokens', type=int, default=768)
    parser.add_argument('--mode', choices=MODES, help='Required for mode/view-conditioned adapters')
    parser.add_argument('--output-view', choices=VIEWS, default='Humanized')
    args = parser.parse_args()
    try:
        system_prompt = inference_system_prompt(args.adapter, args.mode, args.output_view)
    except ValueError as error:
        parser.error(str(error))
    from mlx_lm import load, generate
    model, tokenizer = load('mlx-community/Qwen3-1.7B-4bit', adapter_path=args.adapter)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': args.paragraph}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    print(generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False))


if __name__ == '__main__':
    main()
