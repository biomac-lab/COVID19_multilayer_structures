
# interventions and school capacity of 15%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.15 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.15 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.15 --Tmax 200

# interventions and school capacity of 0%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.0 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.0 --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.0 --Tmax 200

# ## Alternancy 55% - 55%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.55 --school_alternancy True --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.55 --school_alternancy True --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.55 --school_alternancy True --Tmax 200

# ## Alternancy 35%
python run_interventions.py --population 100000 --intervention 0.6 --school_alternancy True --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_alternancy True --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_alternancy True --Tmax 200

# ## Alternancy 15%
python run_interventions.py --population 100000 --intervention 0.6 --school_occupation 0.15 --school_alternancy True --Tmax 200
python run_interventions.py --population 100000 --intervention 0.4 --school_occupation 0.15 --school_alternancy True --Tmax 200
python run_interventions.py --population 100000 --intervention 0.2 --school_occupation 0.15 --school_alternancy True --Tmax 200
