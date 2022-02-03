from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import fire

if __name__ == '__main__':
    fire.Fire(convert_zero_checkpoint_to_fp32_state_dict)