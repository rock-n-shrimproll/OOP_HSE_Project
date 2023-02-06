from base_logic import training
from base_logic.sweep_configs import sweep_config
import wandb
from base_logic.parser import parser


if __name__ == '__main__':
    WANDB_AGENT_DISABLE_FLAPPING = True

    # TODO: login to wandb as function -- now brut forced via terminal
    print('Multimodal EmotionLines Dataset',
          'Encoders Freezing Experiments',
          '',
          'List of preferred encoders:',
          training.encoders_settings,
          '',
          'List of possible freezing settings:',
          training.freezing_settings,
          sep='\n'
          )
    args = parser.parse_args()

    # Setup project
    sweep_config['parameters'].update({'encoder': {'values': [args.encoder_type]},
                                       'freezing': {'values': [args.freezing_type]}})
    sweep_id = wandb.sweep(sweep_config, project=args.project_title)
    # print(sweep_id)
    # Execution
    wandb.agent(sweep_id, training.train_net)
