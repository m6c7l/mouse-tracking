#!/bin/sh
'''which' python3 > /dev/null && exec python3 "$0" "$@" || exec python "$0" "$@"
'''

#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

import lib.env as env_
import lib.utl as utl_
import lib.gui as gui_
import tkinter as tk

# ----------------------------

SENSOR_ARRAY = {'2D':  ({'name': 's.1', 'noise': ( 2,  2), 'bias': (  0,  0), 'interval': 500},
                        {'name': 's.2', 'noise': ( 3,  3), 'bias': (  0,  0), 'interval': 600},
                        {'name': 's.3', 'noise': ( 5,  5), 'bias': (  0,  0), 'interval': 700},
                        {'name': 's.4', 'noise': ( 2,  2), 'bias': (  0,  0), 'interval': 900},
                        {'name': 's.5', 'noise': ( 5,  5), 'bias': (+20,+20), 'interval': 250},
                        {'name': 's.6', 'noise': ( 8,  8), 'bias': (-20,-20), 'interval': 300}),
                        
                '3D':  ({'name': 's.1', 'noise': ( 2,  2,  2), 'bias': (  0,  0,  0), 'interval': 500},
                        {'name': 's.2', 'noise': ( 3,  3,  3), 'bias': (  0,  0,  0), 'interval': 600},
                        {'name': 's.3', 'noise': ( 5,  5,  5), 'bias': (  0,  0,  0), 'interval': 700},
                        {'name': 's.4', 'noise': ( 2,  2,  2), 'bias': (  0,  0,  0), 'interval': 900},
                        {'name': 's.5', 'noise': ( 5,  5,  5), 'bias': (+20,+20,+20), 'interval': 250},
                        {'name': 's.6', 'noise': ( 8,  8,  8), 'bias': (-20,-20,-20), 'interval': 300})}

# ----------------------------
    
if __name__ == '__main__':

    root = gui_.init()
    app = gui_.AppFrame(root, 'virtual testbed: sensor fusion and multi-sensor tracking using standard motion models and recursive Bayesian filters', (1250, 800), '')
    main = gui_.TabFrame(app, tk.TOP)

    # ----------------------------

    tab_T2 = gui_.TabFrame(main, tk.TOP)
    main.add('2D', tab_T2)

    # ----------------------------

    tab_DF = gui_.TabFrame(tab_T2, tk.LEFT)
    tab_T2.add('Fusion', tab_DF)

    # ----------------------------

    tab_DF_KF = gui_.TabFrame(tab_DF, tk.TOP)
    tab_DF.add('KF', tab_DF_KF)

    tab_DF_KF_CV = gui_.FilterFrame(tab_DF_KF, 1, SENSOR_ARRAY['2D'], env_.CV, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DF_KF.add('CV', tab_DF_KF_CV)

    tab_DF_KF_CTL = gui_.FilterFrame(tab_DF_KF, 1, SENSOR_ARRAY['2D'], env_.CT, {'noise': 50, 'dt_millis': 40, 'angle': -60}, env_.EKF, {})
    tab_DF_KF.add('CT/L60', tab_DF_KF_CTL)

    tab_DF_KF_CTR = gui_.FilterFrame(tab_DF_KF, 1, SENSOR_ARRAY['2D'], env_.CT, {'noise': 50, 'dt_millis': 40, 'angle': +30}, env_.EKF, {})
    tab_DF_KF.add('CT/R30', tab_DF_KF_CTR)

    tab_DF_KF_CA = gui_.FilterFrame(tab_DF_KF, 1, SENSOR_ARRAY['2D'], env_.CA, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DF_KF.add('CA', tab_DF_KF_CA)

    tab_DF_KF_CJ = gui_.FilterFrame(tab_DF_KF, 1, SENSOR_ARRAY['2D'], env_.CJ, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DF_KF.add('CJ', tab_DF_KF_CJ)

    tab_DF_KF_BM = gui_.FilterFrame(tab_DF_KF, 1, SENSOR_ARRAY['2D'], env_.BM, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DF_KF.add('Brownian', tab_DF_KF_BM)

    # ----------------------------

    tab_DF_EKF = gui_.TabFrame(tab_DF, tk.TOP)
    tab_DF.add('EKF', tab_DF_EKF)

    tab_DF_EKF_CTRVp = gui_.FilterFrame(tab_DF_EKF, 1, SENSOR_ARRAY['2D'], env_.CTRV, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DF_EKF.add('CTRV', tab_DF_EKF_CTRVp)

    tab_DF_EKF_CTRAp = gui_.FilterFrame(tab_DF_EKF, 1, SENSOR_ARRAY['2D'], env_.CTRA, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DF_EKF.add('CTRA', tab_DF_EKF_CTRAp)

    tab_DF_UKF = gui_.TabFrame(tab_DF, tk.TOP)
    tab_DF.add('UKF', tab_DF_UKF)

    tab_DF_UKF_CTRVp = gui_.FilterFrame(tab_DF_UKF, 1, SENSOR_ARRAY['2D'], env_.CTRV, {'noise': 50, 'dt_millis': 40}, env_.UKF, {})
    tab_DF_UKF.add('CTRV', tab_DF_UKF_CTRVp)

    tab_DF_UKF_CTRAp = gui_.FilterFrame(tab_DF_UKF, 1, SENSOR_ARRAY['2D'], env_.CTRA, {'noise': 50, 'dt_millis': 40}, env_.UKF, {})
    tab_DF_UKF.add('CTRA', tab_DF_UKF_CTRAp)

    # ----------------------------

    tab_DA = gui_.TabFrame(tab_T2, tk.LEFT)
    tab_T2.add('Association', tab_DA)

    # ----------------------------

    tab_DA_KF = gui_.TabFrame(tab_DA, tk.TOP)
    tab_DA.add('KF', tab_DA_KF)

    tab_DA_KF_CV = gui_.FilterFrame(tab_DA_KF, 2, SENSOR_ARRAY['2D'], env_.CV, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DA_KF.add('CV', tab_DA_KF_CV)

    tab_DA_KF_CTL = gui_.FilterFrame(tab_DA_KF, 2, SENSOR_ARRAY['2D'], env_.CT, {'noise': 50, 'dt_millis': 40, 'angle': -60}, env_.EKF, {})
    tab_DA_KF.add('CT/L60', tab_DA_KF_CTL)

    tab_DA_KF_CTR = gui_.FilterFrame(tab_DA_KF, 2, SENSOR_ARRAY['2D'], env_.CT, {'noise': 50, 'dt_millis': 40, 'angle': +30}, env_.EKF, {})
    tab_DA_KF.add('CT/R30', tab_DA_KF_CTR)

    tab_DA_KF_CA = gui_.FilterFrame(tab_DA_KF, 2, SENSOR_ARRAY['2D'], env_.CA, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DA_KF.add('CA', tab_DA_KF_CA)

    tab_DA_KF_CJ = gui_.FilterFrame(tab_DA_KF, 2, SENSOR_ARRAY['2D'], env_.CJ, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DA_KF.add('CJ', tab_DA_KF_CJ)

    tab_DA_KF_BM = gui_.FilterFrame(tab_DA_KF, 2, SENSOR_ARRAY['2D'], env_.BM, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DA_KF.add('Brownian', tab_DA_KF_BM)

    # ----------------------------

    tab_DA_EKF = gui_.TabFrame(tab_DA, tk.TOP)
    tab_DA.add('EKF', tab_DA_EKF)

    tab_DA_EKF_CTRVp = gui_.FilterFrame(tab_DA_EKF, 2, SENSOR_ARRAY['2D'], env_.CTRV, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DA_EKF.add('CTRV', tab_DA_EKF_CTRVp)

    tab_DA_EKF_CTRAp = gui_.FilterFrame(tab_DA_EKF, 2, SENSOR_ARRAY['2D'], env_.CTRA, {'noise': 50, 'dt_millis': 40}, env_.EKF, {})
    tab_DA_EKF.add('CTRA', tab_DA_EKF_CTRAp)

    tab_DA_UKF = gui_.TabFrame(tab_DA, tk.TOP)
    tab_DA.add('UKF', tab_DA_UKF)

    tab_DA_UKF_CTRVp = gui_.FilterFrame(tab_DA_UKF, 2, SENSOR_ARRAY['2D'], env_.CTRV, {'noise': 50, 'dt_millis': 40}, env_.UKF, {})
    tab_DA_UKF.add('CTRV', tab_DA_UKF_CTRVp)

    tab_DA_UKF_CTRAp = gui_.FilterFrame(tab_DA_UKF, 2, SENSOR_ARRAY['2D'], env_.CTRA, {'noise': 50, 'dt_millis': 40}, env_.UKF, {})
    tab_DA_UKF.add('CTRA', tab_DA_UKF_CTRAp)

    # ----------------------------

    tab_T3 = gui_.TabFrame(main, tk.TOP)
    main.add('3D', tab_T3)

    # ----------------------------

    tab_DF3 = gui_.TabFrame(tab_T3, tk.TOP)
    tab_T3.add('Fusion', tab_DF3)

    # ----------------------------

    tab_DF3_CV = gui_.FilterFrame(tab_DF3, 1, SENSOR_ARRAY['3D'], env_.CV, {'dimension': 3, 'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DF3.add('CV (6-state)', tab_DF3_CV)

    tab_DF3_V = gui_.FilterFrame(tab_DF3, 1, SENSOR_ARRAY['3D'], env_.CYRPRV, {'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DF3.add('CYRPRV (8-state)', tab_DF3_V)

    tab_DF3_CA = gui_.FilterFrame(tab_DF3, 1, SENSOR_ARRAY['3D'], env_.CA, {'dimension': 3, 'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DF3.add('CA (9-state)', tab_DF3_CA)

    tab_DF3_A = gui_.FilterFrame(tab_DF3, 1, SENSOR_ARRAY['3D'], env_.CYRPRA, {'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DF3.add('CYRPRA (9-state)', tab_DF3_A)

    tab_DF3_CJ = gui_.FilterFrame(tab_DF3, 1, SENSOR_ARRAY['3D'], env_.CJ, {'dimension': 3, 'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DF3.add('CJ (12-state)', tab_DF3_CJ)

    # ----------------------------

    tab_DA3 = gui_.TabFrame(tab_T3, tk.TOP)
    tab_T3.add('Association', tab_DA3)

    # ----------------------------

    tab_DA3_CV = gui_.FilterFrame(tab_DA3, 2, SENSOR_ARRAY['3D'], env_.CV, {'dimension': 3, 'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DA3.add('CV (6-state)', tab_DA3_CV)

    tab_DA3_V = gui_.FilterFrame(tab_DA3, 2, SENSOR_ARRAY['3D'], env_.CYRPRV, {'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DA3.add('CYRPRV (8-state)', tab_DA3_V)

    tab_DA3_CA = gui_.FilterFrame(tab_DA3, 2, SENSOR_ARRAY['3D'], env_.CA, {'dimension': 3, 'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DA3.add('CA (9-state)', tab_DA3_CA)

    tab_DA3_A = gui_.FilterFrame(tab_DA3, 2, SENSOR_ARRAY['3D'], env_.CYRPRA, {'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DA3.add('CYRPRA (9-state)', tab_DA3_A)

    tab_DA3_CJ = gui_.FilterFrame(tab_DA3, 2, SENSOR_ARRAY['3D'], env_.CJ, {'dimension': 3, 'noise': 10, 'dt_millis': 40}, env_.UKF, {})
    tab_DA3.add('CJ (12-state)', tab_DA3_CJ)

    # ----------------------------

    tk.mainloop()
