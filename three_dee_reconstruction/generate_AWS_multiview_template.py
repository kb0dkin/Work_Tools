import argparse
import os, sys

# create template for an AWS labeling job
def generate_AWS_template(keypoints:list, project_dir):

	# with open(config) as f:
	data = dict()
	data['labels'] = keypoints
	data['header'] = 'Label body parts of mouse in each view'
	data['short_instructions'] = '''Label each body part in as many views as you can. <img src="https://test-3d-template.s3.us-east-2.amazonaws.com/aws_food_body_3D_short.png" width="100%" >
									<br> <b> Nose </b> upper tip of nose
									<br> <b> Throat </b> bottom center of throat, right behind the jaw
									<br> <b> Ears </b> middle front edge
									<br> <b> Spine Center </b> upper center of back
									<br> <b> Knees </b> front curve of each leg
									<br> <b> Paws </b> center of knuckles on each paw
									<br> <b> Tail Base </b> where tail meets the spine
									<br> <b> Tail Mid </b> halfway along length of tail
									<br> <b> Tail Tip </b> end of tail'''
	data['full_instructions'] = '''
  		<img src="https://test-3d-template.s3.us-east-2.amazonaws.com/aws_food_body_3D.png" width="100%" >	
	'''
	data['num_view'] = 5 # number of views
	data['view_bounds'] = [[0,360,640,720],[640,0,1280,360],[640,360,1280,720],[640,720,1280,1080],[1280,360,1920,720]]

	f = open(os.path.join(project_dir, 'annotation_interface.template'), 'w')

	message = """
	<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

	<crowd-form>
  	<div id="errorBox">
  	</div>
    	<crowd-keypoint
        	src="{{{{ task.input.source_ref | grant_read_access }}}}"
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
		var img = document.body.getElementsByTagName("img");
		var i = 0;
		while ( i < img_length ) {{
			img[i].setAttribute("pagespeed_no_transform","");
			i++;
		}}

    	var num_view = {num_view} // this is currently setup as a python f-string
		const view_bounds = {{{{ task.input.view_bounds }}}} // does it work with a straight list?

    	// create a submission callback
    	document.querySelector('crowd-form').onsubmit = function(e) {{
        	const keypoints = document.querySelector('crowd-keypoint').value.keypoints || document.querySelector('crowd-keypoint')._submittableValue.keypoints;
        	const labels = keypoints.map(function(p) {{
        	return p.label;
        	}});

    	// look if a keypoint is within an array bounds -- xyxy
    	function within(label, bounds) {{
        	return label.x > bounds[1] && label.x < bounds[3] && label.y > bounds[0] && label.y < bounds[2];
    	}}

		// check for multiple keypoints within each boundary
    	for (var ii = 0; ii < num_view; ii++) {{
        	labelList = [];
        	Object.entries(keypoints).forEach(entry => {{
            	if (within(entry[1], view_bounds[ii])) {{
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
