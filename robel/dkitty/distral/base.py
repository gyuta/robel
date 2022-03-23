from robel.dkitty.walk import DKittyWalkFixed

# walkタスクそのまま
# heading の情報はorientのものを使ってもよいかも？ eg. facing
OBSERVATION_KEYS = {
    'root_pos',
    'root_euler',
    'kitty_qpos',
    'root_vel',
    'root_angular_vel',
    'kitty_qvel',
    'last_action',
    'upright',
    'heading',
    'target_error',
}

class Base(DKittyWalkFixed):
    """ 複数のタスクのベースとなるクラス

    これを継承してタスクに応じたリワードを与える。
    将来的にはタスクごとに異なる模倣軌道を与えるようにすべきか。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, falling_reward=-2500,**kwargs)
        # super().__init__(*args, falling_reward=-500,**kwargs)