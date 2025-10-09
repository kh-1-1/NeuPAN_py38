from neupan import neupan

if __name__ == '__main__':

    neupan_planner = neupan.init_from_yaml('diff_learned_robot.yaml')
    neupan_planner.train_dune()