"""
Generates synthetic weather images using Stable Diffusion v1.5.
Requires: pip install diffusers transformers accelerate torch
"""

import argparse
from pathlib import Path

import torch

PROMPTS: dict[str, list[str]] = {
    'clear': [
        'outdoor scene with clear blue sky, photorealistic, bright sunlight, sharp detail',
        'sunny day in a park, clear weather, photographic quality, blue sky',
        'clear sky suburban street, bright sunshine, photorealistic, no clouds',
    ],
    'cloudy': [
        'outdoor scene with overcast grey sky, photorealistic, diffuse light, no shadows',
        'cloudy day in a city street, photographic quality, thick cloud cover',
        'suburban neighborhood under heavy cloud cover, photorealistic, grey sky',
    ],
    'foggy': [
        'rural highway in heavy fog at dusk, photographic, low contrast, misty atmosphere',
        'foggy forest path, dense mist, low visibility, photorealistic',
        'city street in thick fog, photographic quality, low contrast, obscured details',
    ],
    'rainy': [
        'heavy rain on a city street, wet pavement, rain streaks, photographic',
        'rainy day outdoor scene, streaking raindrops, blurred background, photorealistic',
        'downpour in a suburban neighborhood, wet road, rain streaks, photographic',
    ],
    'snowy': [
        'snowy suburban street with overcast sky, photographic quality, white ground',
        'heavy snowfall outdoor scene, white coverage, diffuse light, photorealistic',
        'snow covered park on a winter day, diffuse lighting, photographic',
    ],
}

NEGATIVE_PROMPT = (
    'cartoon, illustration, painting, drawing, anime, unrealistic, '
    'blurry, low quality, artifacts, watermark, text'
)


def generate(args):
    from diffusers import StableDiffusionPipeline  # local import so script is importable

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    print(f'Loading Stable Diffusion v1.5 on {device} ({dtype}) ...')

    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    out_root = Path(args.output_dir)
    total = 0

    for cls, prompts in PROMPTS.items():
        cls_dir = out_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(cls_dir.glob('*.png')))  # continue numbering if re-run

        for prompt in prompts:
            print(f'  [{cls}] "{prompt[:70]}"')
            for _ in range(args.per_prompt):
                result = pipe(
                    prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=args.steps,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                )
                result.images[0].save(cls_dir / f'syn_{idx:04d}.png')
                idx += 1
                total += 1

        print(f'  -> {idx} images in {cls_dir}')

    print(f'\nDone. {total} synthetic images saved to {out_root}/')
    print('Re-run prepare_data.py to update splits.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic weather images')
    parser.add_argument('--output-dir', default='data/synthetic')
    parser.add_argument('--per-prompt', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    generate(parser.parse_args())
