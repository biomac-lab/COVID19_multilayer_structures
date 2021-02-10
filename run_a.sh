# run no intervention
python run_dynamical_layers.py --population 100000

# interventions and school capacity of 100%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 1.0
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 1.0
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 1.0

# interventions and school capacity of 55%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.55
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.55
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.55

# # interventions and school capacity of 35%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.35
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.35
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.35

# interventions and school capacity of 25%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.25
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.25
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.25
