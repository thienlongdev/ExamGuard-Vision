import math
try:
    from tools.dataset.preflight_acquisition_v2 import strict_box, supported_classes
except ImportError:
    from scripts.preflight_acquisition_v2 import strict_box, supported_classes
import yaml
from pathlib import Path


def test_nonexistent_cctv_classes_cannot_support_taxonomy():
    mapping = {'datasets': {'cctv': {'source_classes': {'Stand': {'status':'approved','target':'stand'}}}}}
    assert supported_classes(mapping, []) == set()


def test_only_physically_present_approved_classes_support_taxonomy():
    mapping = {'datasets': {'source': {'source_classes': {
        'write': {'status':'approved','target':'normal'},
        'phone': {'status':'needs_review','target':'use_phone'},
        'Stand': {'status':'approved','target':'stand'}}}}}
    rows = [dict(source_dataset='source',source_class_name=name,annotation_count=count)
            for name,count in [('write',2),('phone',5),('Stand',0)]]
    assert supported_classes(mapping,rows) == {'normal'}


def test_strict_bbox_rejects_nonfinite_and_outside_edges():
    assert strict_box([.5,.5,.2,.2])
    assert not strict_box([math.nan,.5,.2,.2])
    assert not strict_box([.5,.5,math.inf,.2])
    assert not strict_box([0,.5,.1,.2])
    assert not strict_box([.5,.5,0,.2])


def test_phone_objects_never_map_to_behavior_in_acquisition_decisions():
    decisions = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/acquisition_v2_decisions.yaml').read_text())
    phone = decisions['dataset_decisions']['phone_candidate_student_behaviour']['source_classes']
    assert phone['phone']['target'] is None
    assert phone['Using_phone']['target'] is None
    assert decisions['phone_strategy'] == 'coco_object_association_temporal'
    assert 'use_phone' not in decisions['supported_behavior_classes']
    assert not decisions['sanity_training_allowed']
    assert not decisions['lean_enabled']
