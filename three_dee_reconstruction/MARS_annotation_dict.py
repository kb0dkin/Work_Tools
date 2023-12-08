dataset_info = dict(
    dataset_name='open_field',
    paper_info=dict(
        author='Arin',
        title='AWS_tags for open field',
        container='test_container',
        year='2023',
        homepage='https://github.com/neuroethology/MARS_developer/',
    ),
    keypoint_info={
        0:
        dict(
            name='nose', 
            id=0, 
            color=[0, 0, 255], 
            type='lower', 
            swap=''),
        
        1:
        dict(
            name='throat', 
            id=1, 
            color=[0, 0, 255], 
            type='lower', 
            swap=''),
        
        2:
        dict(
            name='body center', 
            id=2, 
            color=[0, 0, 255], 
            type='lower', 
            swap=''),
        
        3:
        dict(
            name='right ear',
            id=3,
            color=[255, 128, 0],
            type='lower',
            swap='left ear'),
        
        4:
        dict(
            name='left ear', 
            id=4, 
            color=[51, 153, 255], 
            type='lower', 
            swap='right ear'),
        
        5:
        dict(
            name='right hip', 
            id=5, 
            color=[51, 153, 255], 
            type='lower', 
            swap='left hip'),
        
        6:
        dict(
            name='left hip',
            id=5,
            color=[51, 153, 255],
            type='lower',
            swap='right hip'),
        
        7:
        dict(
            name='tail base',
            id=7,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        8:
        dict(
            name='tail mid',
            id=8,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        9:
        dict(
            name='tail tip',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap=''),
    },
    
    skeleton_info={
        0: dict(link=('nose', 'right ear'), id=0, color=[0, 0, 255]),
        1: dict(link=('nose', 'left ear'), id=1, color=[0, 0, 255]),
        2: dict(link=('right ear', 'throat'), id=2, color=[0, 0, 255]),
        3: dict(link=('left ear', 'throat'), id=3, color=[0, 0, 255]),
        4: dict(link=('throat', 'body center'), id=4, color=[0, 255, 0]),
        5: dict(link=('body center', 'right hip'), id=5, color=[0, 255, 255]),
        6: dict(link=('body center', 'left hip'), id=6, color=[0, 255, 255]),
        7: dict(link=('right hip', 'tail base'), id=6, color=[0, 255, 255]),
        8: dict(link=('left hip', 'tail base'), id=7, color=[6, 156, 250]),
        9: dict(link=('tail base', 'tail mid'), id=8, color=[6, 156, 250]),
        10: dict(link=('tail mid', 'tail tip'), id=9, color=[6, 156, 250]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5
    ],
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072
    ])
