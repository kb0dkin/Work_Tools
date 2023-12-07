import argparse
import yaml
import os, sys

# create template for an AWS labeling job
def generate_AWS_template(keypoints:list):

	# with open(config) as f:
	# 	data = yaml.load(f, Loader=yaml.FullLoader)
	# data['labels'] = [n + " " + data['species'] + " " + k for n in data['animal_names'] for k in data['keypoints']]
	data = dict()
	data['labels'] = keypoints
	data['header'] = 'Label body parts of mouse in each view'
	data['short_instructions'] = '''Label each body part in as many views as you can. Please view full instructions for examples'''
	data['full_instructions'] = 'Lorem Ipsum'
	data['num_img'] = 5
	data['img_bounds'] = [[0,360,640,720],[640,0,1280,360],[640,360,1280,720],[640,720,1280,1080],[1280,360,1920,720]]

	# project_dir = os.path.dirname(config)
	project_dir = os.path.abspath('/home/klb807/Documents/')
	f = open(os.path.join(project_dir, 'annotation_interface.template'), 'w')

	message = """
	<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

	<crowd-form>
  	<div id="errorBox">
  	</div>
    	<crowd-keypoint
        	src="{{{{ task.input.taskObject | grant_read_access }}}}"
        	labels="{labels}"
        	header="{header}"
        	name="annotatedResult">

          	<short-instructions>
          	{short_instructions}
          	</short-instructions>

          	<full-instructions header="Instructions" >
          	{full_instructions}
          	</full-instructions>

    	</crowd-keypoint>
	<br></br>
	<!--Additional instructions/sample images could go here-->

	</crowd-form>


	<!----------------------------------------Script to ensure each body part is annotated exactly N times------------------------------------------------>
	<script>
    	var num_img = {num_img} // this is currently setup as a python f-string
		const img_bounds = {img_bounds} // does it work with a straight list?

    	// create a submission callback
    	document.querySelector('crowd-form').onsubmit = function(e) {{
        	const keypoints = document.querySelector('crowd-keypoint').value.keypoints || document.querySelector('crowd-keypoint')._submittableValue.keypoints;
        	const labels = keypoints.map(function(p) {{
        	return p.label;
        	}});

    	// look if a keypoint is within an array bounds -- xyxy
    	function within(label, bounds) {{
        	return label.x > bounds[0] && label.x < bounds[2] && label.y > bounds[1] && label.y > bounds[3];
    	}}

    	for (var ii = 0; ii < num_img; ii++) {{
        	labelList = [];
        	Object.entries(keypoints).forEach(entry => {{
            	if (within(entry[1], img_bounds[ii])) {{
            	if (labelList.includes(entry[1].label)){{
                	e.preventDefault();
                	errorBox.innerHTML = '<crowd-alert type="error">'+ entry[1].label + ' is tagged multiple times in view '+ ii +'</crowd-alert>';
                	errorBox.scrollIntoView();  
            	}} else {{
                	labelList = labelList.concat(entry[1].label)
            	}}
          
            	}}
        	}})
    	}}

	}};
	</script>

	<style>
	img {{
  	display: block;
  	margin-left: auto;
  	margin-right: auto;
	}}
	</style>
	
	""".format(**data)

	f.write(message)
	f.close()


		# const img_bounds = [[0,360,640,720],[640,0,1280,360],[640,360,1280,720],[640,720,1280,1080],[1280,360,1920,720]]
    	# const img_bounds = [{img_bounds[0,:]},{img_bounds[1,:]},{img_bounds[2,:]},{img_bounds[3,:]}] // boundary of each of the views in image xyxy