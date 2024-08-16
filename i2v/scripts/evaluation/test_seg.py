import sys

sys.path.append('/root/godmodeai/god-mode-ai-server/src/bg_tasks/')

from segment_animation import segment_gif
from segment_animation import gif_to_sprite_sheet



segment_gif("./test_anim.gif", "/tmp/test_anim_seg", "/tmp/test_anim_seg.gif")

ret = gif_to_sprite_sheet("/tmp/test_anim_seg.gif")

print(f"gened: {ret}")