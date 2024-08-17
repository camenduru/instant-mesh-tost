import torch
import numpy as np
import rembg
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from src.utils.infer_util import remove_background, resize_foreground

from torchvision.transforms import v2
from omegaconf import OmegaConf
from einops import repeat
import tempfile
from tqdm import tqdm
import imageio

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (FOV_to_intrinsics, get_zero123plus_input_cameras,get_circular_camera_poses,)
from src.utils.mesh_util import save_obj, save_obj_with_mtl

import os, json, requests, runpod

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

def preprocess(input_image, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    return input_image

def generate_mvs(input_image, sample_steps, sample_seed, pipeline, device):
    seed_everything(sample_seed)
    generator = torch.Generator(device=device)
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps, 
        generator=generator,
    ).images[0]
    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())
    return z123_image, show_image

def images_to_video(images, output_path, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')

def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

def make_mesh(mesh_fpath, planes, model, infer_config, export_texmap):
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_vis_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
    with torch.no_grad():
        mesh_out = model.extract_mesh(planes, use_texture_map=export_texmap, **infer_config,)
        if export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_fpath,
            )
            print(f"Mesh with texmap saved to {mesh_fpath}")
        else:
            vertices, faces, vertex_colors = mesh_out
            vertices = vertices[:, [1, 2, 0]]
            vertices[:, -1] *= -1
            faces = faces[:, [2, 1, 0]]
            save_obj(vertices, faces, vertex_colors, mesh_fpath)
            print(f"Mesh saved to {mesh_fpath}")
    return mesh_fpath

def make3d(images, model, device, IS_FLEXICUBES, infer_config, export_video, export_texmap):
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device)
    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)
    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")
    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384
        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(planes, render_cameras[:, i:i+chunk_size], render_size=render_size,)['img']
            else:
                frame = model.synthesizer(planes, cameras=render_cameras[:, i:i+chunk_size],render_size=render_size,)['images_rgb']
            frames.append(frame)
        frames = torch.cat(frames, dim=1)
        if export_video:
            images_to_video(frames[0], video_fpath, fps=30,)
            print(f"Video saved to {video_fpath}")
    mesh_fpath = make_mesh(mesh_fpath, planes, model, infer_config, export_texmap)
    if export_video:
        return video_fpath, mesh_fpath
    else:
        return mesh_fpath

@torch.inference_mode()
def generate(input):
    values = input["input"]
    input_image = values['input_image']
    sample_steps = values['sample_steps']
    seed = values['seed']
    remove_background = True
    export_video = True
    export_texmap = True

    input_image = load_image(input_image)
    processed_image = preprocess(input_image, remove_background)

    model = None
    torch.cuda.empty_cache()
    pipeline = DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="zero123plus",torch_dtype=torch.float16,)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda')
    pipeline = pipeline.to(device)
    seed_everything(0)
    mv_images, mv_show_images = generate_mvs(processed_image, sample_steps, seed, pipeline, device)

    pipeline = None
    torch.cuda.empty_cache()
    config_path = 'configs/instant-mesh-base.yaml'
    config = OmegaConf.load(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_base.ckpt", repo_type="model")
    model = instantiate_from_config(model_config)
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda')
    model = model.to(device)
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()

    output_video, output_model_obj = make3d(mv_images, model, device, IS_FLEXICUBES, infer_config, export_video, export_texmap)
    mesh_basename = os.path.splitext(output_model_obj)[0]
    
    result = [output_video, [output_model_obj, mesh_basename+'.mtl', mesh_basename+'.png']]
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        if(notify_uri == "notify_uri"):
            notify_uri = os.getenv('com_camenduru_notify_uri')
            notify_token = os.getenv('com_camenduru_notify_token')
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result[0])
        with open(result[0], "rb") as file:
            files = {default_filename: file.read()}
        for path in result[1]:
            filename = os.path.basename(path)
            with open(path, "rb") as file:
                files[filename] = file.read()
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_urls = [attachment['url'] for attachment in response.json()['attachments']]
        notify_payload = {"jobId": job_id, "result": str(result_urls), "status": "DONE"}
        requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": str(result_urls), "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result[0]):
            os.remove(result[0])
        if os.path.exists(result[1][0]):
            os.remove(result[1][0])
        if os.path.exists(result[1][1]):
            os.remove(result[1][1])
        if os.path.exists(result[1][2]):
            os.remove(result[1][2])

runpod.serverless.start({"handler": generate})