from neupan import neupan

if __name__ == '__main__':

    # Use an existing baseline config for diff robot (ObsPointNet front)
    planner = neupan.init_from_yaml('dune_train_diff_CLARABEL.yaml')
    planner.train_dune()

