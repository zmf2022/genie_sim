# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

INSTRUCTION_TEMPLATE = {
    "pick_billiards_color": "{[SIDE]: left/right} arm picks up the {[COLOR]: red/green/blue/..., color of the object in the scene_info} billiard on the table",
    "pick_block_color": "{[SIDE]: left/right} arm picks up the {[COLOR]: red/green/blue/..., color of the object in the scene_info} block on the table",
    "pick_block_shape": "{[SIDE]: left/right} arm picks up the {[SHAPE]: cube/sphere/cylinder/..., shape of the object in the scene_info} block on the table",
    "pick_block_size": "{[SIDE]: left/right} arm picks up the {[SIZE]: smallest/biggest/..., relative size of the object in the scene_info} block on the table",
    "pick_block_number": "{[SIDE]: left/right} arm picks up the building block with number {[NUMBER]: 0/1/2/...} on the table",
    "pick_object_type": "{[SIDE]: left/right} arm picks up the {[TYPE]: fruit/toy/food/beverage/..., type of the object in the scene_info} on the table",
    "pick_specific_object": "{[SIDE]: left/right} arm picks up the {[SEMANTIC]: pen/cup/fruit/..., semantic of the object in the scene_info} on the table",
    "pick_follow_logic_or": "VLM|Picks up the {[DESCRIPTION]: A or B, description of the object in the scene_info} item on the table",
    "pick_common_sense": "{[SIDE]: left/right} arm picks up the object that {[COMMON_SENSE]: use common sense description sentences to describe the feature of the object in the scene_info without mentioning name} on the table",
    "place_block_into_box": "{[SIDE]: left/right} arm picks up the {[COLOR]: red/green/blue/..., color of the object in the scene_info} {[SHAPE]: cube/sphere/cylinder/..., shape of the object in the scene_info} block on the table, places it into the {[SHAPE]: cube/sphere/cylinder/..., shape of the object in the scene_info} hole of the building block box",
    "place_object_into_box_color": "{[SIDE]: left/right tag of the apple in scene_info} arm picks up the apple on the table, places it into the {[COLOR]: color of the storage box in scene_info, every color should be used to generate instruction} storage box",
    "straighten_object": "{[SIDE]: left/right} arm picks up the beverage on the table,  straightens the beverage and puts it back in its original position",
    "clean_the_desktop": "VLM|clean up the desktop",
    "sort_fruit": "VLM|Sort the fruit into the corresponding box",
    "open_door": "VLM|Right arm opens the door, push it open",
    "hold_pot": "VLM|Both arms hold the pot and place it on the stove",
    "straighten_object": "VLM|Straighten the {[SEMANTIC]: the semantic_name of beverage in the scene_info} on the table",
    "bimanual_chip_handover": "Pick up the chip bag with the nearest hand, pass it to the other hand, and keep it upright",
}
