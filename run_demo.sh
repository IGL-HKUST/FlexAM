# Motion transfer
python demo.py \
    --prompt "A bright orange fox with a fluffy coat and a white muzzle stands on a bed of fallen autumn leaves, enjoying a warm moment of contentment. Throughout the motion, the fox keeps its mouth fully closed and its facial expression unchanged, with no sudden shifts or ghosting artifacts. After a gentle stretch, it slowly settles back onto its hind legs and lifts its tail in a neat upward curl, all while maintaining a clean and stable look in the frame." \
    --input_path /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/fox.mp4 \
    --repaint=/hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/motiontransfer/fox_motion.png \
    --video_length=97 \
    --sample_size 512 896\
    --generate_type='full_edit' \
    --output_dir output/foxmotion_10_12134 \
    --density 10 \
    --checkpoint_path='/hpc2hdd/home/msheng758/projects/dasv2/models/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM' \

# # foreground edit
# python demo.py \
#     --prompt "A gray cat with bright yellow eyes walks cautiously across a snow-covered slope, its dark fur standing out against the white snow. The background features a forest of bare branches with scattered orange leaves, creating a strong contrast with the wintry foreground. The cat’s focused expression and low, deliberate posture convey alertness as it moves quietly through the cold landscape, surrounded by the stillness of early winter" \
#     --input_path /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/fox.mp4 \
#     --repaint=/hpc2hdd/home/msheng758/projects/hua/das/ComfyUI/output/1120/fgbg/condition_dasv2/comfyui_00001.png \
#     --mask_path=/hpc2hdd/home/msheng758/projects/hua/das/ComfyUI/output/1120/fgbg/mask_dasv2/comfyui_00001.mp4 \
#     --video_length=97 \
#     --sample_size 512 896\
#     --generate_type='foreground_edit' \
#     --dilation_pixels=600 \
#     --output_dir output/fox_fg_cat_15density \
#     --density 15 \
#     --checkpoint_path='/hpc2hdd/home/msheng758/projects/dasv2/models/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM' \
    

# # background edit
# python demo.py \
#     --prompt "A fluffy white fox stands in a vibrant spring meadow filled with colorful flowers, including daisies, tulips, and small wild blossoms. The bright green grass contrasts with the fox’s soft white fur. The fox’s eyes are gently narrowed, giving it a calm, content expression as it enjoys the peaceful spring surroundings." \
#     --input_path /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/fox.mp4 \
#     --repaint=/hpc2hdd/home/msheng758/projects/hua/das/ComfyUI/output/1109/fgbg/condition_dasv2/comfyui_00018.png \
#     --mask_path=/hpc2hdd/home/msheng758/projects/hua/das/ComfyUI/output/1120/fgbg/mask_dasv2/comfyui_00001.mp4 \
#     --video_length=97 \
#     --sample_size 512 896\
#     --generate_type='background_edit' \
#     --output_dir output/fox_bg_15density \
#     --density 15 \
#     --checkpoint_path='/hpc2hdd/home/msheng758/projects/dasv2/models/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM' \
    

# # camara ctrl, pose txt
# python demo.py \
#     --prompt "A white fox stands poised on a snow-covered slope, its silvery coat blending seamlessly with the wintry landscape, its pink nose twitching and pointed pink ears adding soft color to the monochromatic scene. Behind it, a forest of autumnal orange leaves offers a gentle contrast to the snow and the fox’s pale fur. As the camera moves, the fox retains its natural, undistorted form—every detail of its sleek coat, distinct facial features, and elegant proportions remains intact with no warping, deformation, or unnatural extra appendages. The fox arches its back in a subtle stretch, then extends into a relaxed, full-body stretch before slowly sitting back onto its hind legs; its tail naturally unfurls and gently drifts down to rest softly on the snow-covered ground, maintaining a single, coherent form throughout. The frame remains clean and visually stable, emphasizing the fox’s graceful, natural movements." \
#     --input_path /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/fox.mp4 \
#     --camera_motion "path" \
#     --pose_file /hpc2hdd/home/msheng758/projects/VideoX-Fun/asset/Pan_Right_Up.txt \
#     --tracking_method DELTA \
#     --override_extrinsics append \
#     --output_dir output/fox_pan_right_up_density15_new_prompt \
#     --sample_size 512 896 \
#     --video_length=97 \
#     --density 15 \
#     --checkpoint_path='/hpc2hdd/home/msheng758/projects/dasv2/models/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM' \
    

# # camara ctrl, pi3
# python demo.py \
#     --prompt "A white fox stands poised on a snow-covered slope, its silvery coat blending seamlessly with the wintry landscape, its pink nose twitching and pointed pink ears adding soft color to the monochromatic scene. Behind it, a forest of autumnal orange leaves offers a gentle contrast to the snow and the fox’s pale fur. As the camera moves, the fox retains its natural, undistorted form—every detail of its sleek coat, distinct facial features, and elegant proportions remains intact with no warping or deformation. The fox arches its back in a subtle stretch, then extends into a relaxed, full-body stretch before slowly sitting back onto its hind legs and lifting its tail in a neat, graceful curl, all while keeping the frame clean and visually stable." \
#     --input_path /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/fox.mp4 \
#     --camera_motion "path" \
#     --pose_file /hpc2hdd/home/msheng758/projects/dasv2/outputs/camera_ctrl_pi3/zoomin_fox_copy/result.mp4 \
#     --tracking_method DELTA \
#     --override_extrinsics append \
#     --output_dir output/camera_ctrl_pi3_zoomin_fox2 \
#     --sample_size 512 896 \
#     --video_length=81 \
#     --density 15 \
#     --checkpoint_path='/hpc2hdd/home/msheng758/projects/dasv2/models/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM' \
    

# # camara ctrl
# python demo.py \
#     --prompt "A black and tan dachshund with expressive eyes and floppy ears is first seen peeking from behind a white door frame, wearing a turquoise shirt with white polka dots, before retreating and tucking its head back behind the door frame. The dog's body language suggests curiosity and eagerness, possibly indicating a desire for interaction or exploration. The scene is set against a simple backdrop, with a blurred background that hints at a domestic setting. The dachshund's attire and the soft lighting contribute to a warm and inviting atmosphere, emphasizing the dog's endearing features. The dog's playful sequence of peeking and then pulling back, paired with the cozy indoor setting, suggests a moment of quiet anticipation and charming shyness.Captured in a smooth spiral camera movement that gently orbits around—slowly circling around while maintaining a steady distance, shifting perspective subtly to showcase different angles. The spiral motion is fluid and unobtrusive, adding a dynamic yet serene rhythm to the scene." \
#     --input_path /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/dog3.mp4 \
#     --camera_motion "spiral 6" \
#     --tracking_method DELTA \
#     --override_extrinsics append \
#     --output_dir output/dog3_spiral_limit_density5 \
#     --sample_size 512 896 \
#     --video_length=89 \
#     --density 5 \
#     --checkpoint_path='/hpc2hdd/home/msheng758/projects/dasv2/models/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM' \
    

# # Object manipulation
# python demo.py \
#     --prompt "A beautiful white arctic fox is slowly turning its head in a smooth, natural motion, with no sign of distortion or unnatural movement. As it turns, every part of its head, from its sharp eyes to its fluffy white furry face, stays perfectly normal and well-formed, without any flattening, squashing, or deformation. The turn is a single-direction, graceful motion, keeping the fox's elegant appearance intact throughout the entire process." \
#     --input_path /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/videos/fox_first_frame.mp4 \
#     --object_motion roll_right \
#     --object_mask /hpc2hdd/JH_DATA/share/msheng758/PrivateShareGroup/msheng758_das/das2_demo/1119/mask/fox.png \
#     --tracking_method DELTA \
#     --sample_size 512 896 \
#     --video_length=49 \
#     --density 30 \
#     --output_dir output/fox_roll_right_density30_motion_magnitude50_896 \
#     --checkpoint_path='/hpc2hdd/home/msheng758/projects/dasv2/models/Diffusion_Transformer/Wan2.2-Fun-5B-FLEXAM' \
    