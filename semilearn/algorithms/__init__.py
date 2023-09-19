from .fixmatch import FixMatch
from .openmatch import OpenMatch
from .iomatch import IOMatch

# if any new alg., please append the dict
name2alg = {
    "fixmatch": FixMatch,
    'openmatch': OpenMatch,
    "iomatch": IOMatch
}


def get_algorithm(args, net_builder, tb_log, logger):
    try:
        alg = name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    except KeyError as e:
        print(f'Unknown algorithm: {str(e)}')
