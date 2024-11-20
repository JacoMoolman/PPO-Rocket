import neat
import pickle
from GetMoonGame import MoonLanderGame
from functools import partial

def eval_genomes(genomes, config, generation):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = MoonLanderGame(net)
        genome.fitness = game.run_genome(genome, generation)  # Pass the generation number to run_genome

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(partial(eval_genomes, generation=p.generation), 50000)

    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    config_path = 'config-feedforward.txt'
    run(config_path)
