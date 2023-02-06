import argparse

parser = argparse.ArgumentParser(description='Experiments with Freezing Encoders')
parser.add_argument('project_title', type=str,
                    help='Project title for sweep_id'
                    )
parser.add_argument('encoder_type', type=str, choices=['bert-base-uncased',
                                                       'bert-base-cased-conv',
                                                       'roberta-base',
                                                       'roberta-base-conv',
                                                       'xlnet-base-cased'],
                    help='Encoder to experiment with'
                    )
parser.add_argument('freezing_type', type=str, default='none', choices=['none',
                                                                        '2_frozen',
                                                                        '4_frozen',
                                                                        'full'],
                    help='Number of frozen layers'
                    )
