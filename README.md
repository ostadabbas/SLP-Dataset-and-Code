#  SLP datasets 
This is a demonstration work of SLP datasets by estimation human 2D/3D pose from multiple modalities. 
SLP interface is provided for quick data I/O of SLP dataset. 
Several state of the art methods are included for test purpose. 

## dependencies 
scipy

matplotlib

opencv 
 
torch

torchvision 

scikit-image

scipy

visdom

jsonpatch(visdom need it)

dominate (if using visualizer, otherwise commented it out)

## file structure 

    trainNm/	module_dump: save weights 

		log:	record training loss and test loss  separately, we can only check test 
		
		result:	test result of the metrics, predictions (json, npy) 
		
		vis:	the visualization result 
		
			division(test, train)/	/testSet	/2d
			
							/3d
							
					        /hist		
					
							…
							
		web: 	display on server 
