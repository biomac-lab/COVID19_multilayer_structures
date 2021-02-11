# run no intervention
# python run_dynamical_layers.py --population 100000 --Tmax 200

# interventions and school capacity of 100%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 1.0 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 1.0 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 1.0 --Tmax 200

# interventions and school capacity of 55%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.55 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.55 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.55 --Tmax 200

# # interventions and school capacity of 35%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.35 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.35 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.35 --Tmax 200

# interventions and school capacity of 25%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.25 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.25 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.25 --Tmax 200