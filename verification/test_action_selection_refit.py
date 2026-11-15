"""Final action reader, strict ahead geometry and arbitrary-frame history."""
import numpy as np
import pandas as pd
from PRESS import PRESS
from _fixtures import load_test_match
from PRESS.model import ActionSelectionModel
from PRESS.model.action_features import aggregate

def test_installed_native_action_refit_has_the_verified_contract():
    model=ActionSelectionModel.pretrained();a=model.to_dict()
    assert model.model_id=='skillcorner_outfield_action_selection_v2'
    assert model.booster.num_boosted_rounds()==277
    assert a['feature_contract']=='canonical_action_30_v1'
    assert a['training']['n_actions']==16918 and len(a['training']['match_ids'])==10
    assert a['calibration']=={'method':'none'} and not a['training']['smoke']

def test_hypothetical_success_failure_counts_and_mirrored_attack_agree():
    own=np.column_stack([[-10,-5,0,1,2,3,4,5,6,7],np.zeros(10)])
    opp=np.column_stack([[-15,-7,-1,0,1,3,5,7,9,11,15],np.zeros(11)])
    players=np.vstack([own,[0.,0.],opp]);targets=np.array([[-5.,0.],[0.,0.],[5.,0.]])
    q=np.array([.2,.3,.5]);p=np.array([.9,.6,.7]);common=dict(choice=q,completion=p,drive_loss=.2,bcos=10.,
        xt_grid=np.ones((10,6))*.01,pitch_length=105.,pitch_width=68.,match_clock_min=4.,
        score_difference_prefix=0.,completed_canonical_drives=1)
    a=aggregate(bc=np.zeros(2),velocity=np.array([1.,0.]),players=players,targets=targets,direction=1,
        own_lines=np.array([-5.,0.,5.]),opp_lines=np.array([-7.,0.,7.]),**common)
    b=aggregate(bc=np.zeros(2),velocity=np.array([-1.,0.]),players=-players,targets=-targets,direction=-1,
        own_lines=np.array([5.,0.,-5.]),opp_lines=np.array([7.,0.,-7.]),**common)
    assert a==b
    assert a['teammates_ahead_bc']==7 and a['opponents_ahead_bc']==7
    assert np.isclose(a['teammates_ahead_pass_success'],(q*p)@np.array([8,7,2]))
    assert np.isclose(a['teammates_ahead_pass_success']+a['teammates_ahead_pass_failure'],q@np.array([8,7,2]))
    assert np.isclose(a['opponents_ahead_pass_success']+a['opponents_ahead_pass_failure'],q@np.array([9,7,4]))
    assert np.isclose(a['pass_success_mean'],q@p)

def test_raw_unknown_context_and_completed_count_follow_individual_history():
    tracking,teams,actions,_=load_test_match();model=PRESS(tracking,possessions_data=teams,actions_data=actions)
    model.player_possessions_data=pd.DataFrame([dict(frame_start=0,frame_end=29,start_type='unknown')])
    for frame,count,previous in [(4,0,'unknown'),(14,1,'ball_drive'),(24,2,'ball_drive')]:
        context=model.get_frame_context(frame)
        assert context['start_type']=='unknown' and context['previous_action']==previous
        assert context['completed_canonical_drives']==count
