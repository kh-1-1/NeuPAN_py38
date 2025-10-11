from neupan import neupan

if __name__ == '__main__':

    planner = neupan.init_from_yaml('dune_train_diff.yaml')
    planner.train_dune()

