from neupan import neupan

if __name__ == '__main__':

    neupan_planner = neupan.init_from_yaml('acker_flex_pdhg-row-step_se2_robot.yaml')
    neupan_planner.train_dune()