from neupan import neupan

if __name__ == '__main__':

    neupan_planner = neupan.init_from_yaml('dune_train_acker_kkt_tuned.yaml')
    neupan_planner.train_dune()