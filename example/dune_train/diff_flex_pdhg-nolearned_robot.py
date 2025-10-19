from neupan import neupan

if __name__ == '__main__':

    planner = neupan.init_from_yaml('diff_flex_pdhg-nolearned_robot.yaml')
    planner.train_dune()

