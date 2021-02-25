"""
+ Data and calculation basen on 'COVID-19 Aerosol Transmission Estimation', developed by: Prof. Jose L Jimenez
+ Dept. of Chem. and CIRES, University of Colorado-Boulder
+ https://tinyurl.com/covid-estimator
"""

import numpy as np2

# Environmental parameters for Bogotá -----------------------------------------------------------

PRESSURE    = 0.74  # atm
TEMPERATURE = 30    # C 
RELATIVE_HUMIDITY = 0.8     # (1)

DACAY_RATE_VIRUS  = 0.63    # h-1
DEPOSITION_SURFACES = 0.3   # h-1
ADDITIONAL_CONTROLS = 0     # h-1

# People and activity in room parameters -------------------------------------------------------

INFECTED_PEOPLE = 1     # Assumption of infected people in room

QUANTA_EXHATATION_RATE = 25     # Infectious doses (quanta) h-1

# Parameters related to the COVID-19 -----------------------------------------------------------

BREATHING_RATE = 0.005*60   # m3/h

# Functions ------------------------------------------------------------------------------------

def ventilation_rate_per_person(lenght_room,width_room,height_room,ventilation_out,N_people):
    room_volume = lenght_room*width_room*height_room     # m3
    return room_volume*(ventilation_out+ADDITIONAL_CONTROLS)*(1000/3600/N_people)


def net_emmision_rate(exhalation_mask_eff,fraction_people_masks,N_infected=INFECTED_PEOPLE):
    return QUANTA_EXHATATION_RATE*(1-exhalation_mask_eff*fraction_people_masks)*N_infected


def avg_quanta_concentration(lenght_room,width_room,height_room,ventilation_out,
            exhalation_mask_eff,fraction_people_masks,duration_event_h):
    """
    Analytical solution of the box model. Eq (4) in Miller et al. (2020)
    """
    room_volume = lenght_room*width_room*height_room     # m3
    TOTAL_FO_LOSS_RATE = ventilation_out+DACAY_RATE_VIRUS+DEPOSITION_SURFACES+ADDITIONAL_CONTROLS
    net_emmision_rate_val = net_emmision_rate(exhalation_mask_eff,fraction_people_masks)
    return (net_emmision_rate_val/TOTAL_FO_LOSS_RATE/room_volume)*(1-(1/TOTAL_FO_LOSS_RATE/duration_event_h)*(1-np2.exp(-TOTAL_FO_LOSS_RATE*duration_event_h)))


def quanta_inhaled_per_person(lenght_room,width_room,height_room,ventilation_out,
            inhalation_mask_eff,exhalation_mask_eff,fraction_people_masks,duration_event_h,breathing_rate=BREATHING_RATE):
    avg_quanta_concentration_val = avg_quanta_concentration(lenght_room,width_room,height_room,ventilation_out,
                                                            exhalation_mask_eff,fraction_people_masks,duration_event_h)
    return avg_quanta_concentration_val*breathing_rate*duration_event_h*(1-inhalation_mask_eff*fraction_people_masks)


def infection_probability(lenght_room,width_room,height_room,ventilation_out,
            inhalation_mask_eff,exhalation_mask_eff,fraction_people_masks,duration_event_h,breathing_rate=BREATHING_RATE):
    quanta_inhaled_per_person_val = quanta_inhaled_per_person(lenght_room,width_room,height_room,ventilation_out,
            inhalation_mask_eff,exhalation_mask_eff,fraction_people_masks,duration_event_h,breathing_rate=BREATHING_RATE)
    return 1-np2.exp(-quanta_inhaled_per_person_val)