
# Deep Learning based Inter-Modality Image Registration Supervised by Intra-Modality Similarity

 The aim of the project is to reveal the geometric accuracy of tomography images and the soft tissue resolution quality of  MRI images with deep learning-based image registration method.

## Principles of Radiography

  Radiology is the use of x-rays and other imaging methods in medicine for diagnosis and treatment.

### COMPUTED TOMOGRAPHY ( CT )

  By taking many two-dimensional X-ray images of an object from different angles, it is tried to obtain a three-dimensional image of the internal structure of that object.
  +    Creates a detailed and layered picture by scanning the   relevant area or areas of the body with X-r
  +    Gives the closest results geometrically.
  +    Helpful in applications such as drug dosage for disease treatment.

![image](https://user-images.githubusercontent.com/38917811/122679484-be021e00-d1f3-11eb-9f66-ea47d9f5c30e.png)


###  MAGNETIC RESONANCE ( MRI )

Magnetic Resonance Imaging (MRI) is a technique that uses magnetic fields and radio waves to create detailed images of organs and tissues in the body 

 +   High-resolution images of the inside of the body are obtained to diagnose      various problems. 
 +   Most widely used method for disease diagnosis. 

![image](https://user-images.githubusercontent.com/38917811/122679491-c3f7ff00-d1f3-11eb-8bfb-2f6af20d1453.png)


###  DIFFERENCE BETWEEN MRI AND CT
 
 **CT** <br/>
![image](https://user-images.githubusercontent.com/38917811/122679568-12a59900-d1f4-11eb-88f2-1f251e51f8de.png) 
+ High geometric accuracy.
+ Low image resolution. <br/>
 **MR** <br/>
![image](https://user-images.githubusercontent.com/38917811/122679576-1d602e00-d1f4-11eb-8b7f-1d4a5d0e9e4d.png)

###  PROJECT DEVELOPMENT ENVIRONMENT

  Examining the development environment, Anaconda3, Spyder IDE used for the project because Spyder provides more convenience for the data processing, such as data visualization, ITK, Variable Tracking. Keras, Tensorflow libraries and  deep learning modules will be used for the image registration. PyQt library used for the desktop application.

### BASIC IMAGE REGISTRATION

There are four basic steps incluiding Interpolation, Metric, Optimization, Transformation in the project. <br/>

![image](https://user-images.githubusercontent.com/38917811/122679890-59e05980-d1f5-11eb-951c-3009927d5efe.png)

## Interpolation
 
 +   When a point is mapped from one space to another with a transformation, it is generally mapped to a non-grid location. Therefore, interpolation is necessary to evaluate the   image density at the mapped location.
 +  Transformation maps point from the still image coordinate system to the moving image coordinate system. <br/>

![image](https://user-images.githubusercontent.com/38917811/122679768-cad34180-d1f4-11eb-8931-3de5378e9754.png)

### Interpolation Types

  Linear Interpolation :   A method that assumes the density
  varies linearly between grid positions.

  B-Spline Interpolation :   A method for comparing image
  density using the B-spline function.

  Windowed Sinc Interpolation :   A method that performs
  interpolation based on Fourier analysis.

Windowed Sinc Interpolation used in the project

## Metric

 +  This is the most critical part of the registration process.

 + The choice of the metric algorithm is directly determined by the problem to be solved.

 +  We used the histogram-based mutual information method because it is highly recommended in medical registrations.

![image](https://user-images.githubusercontent.com/38917811/122679926-92803300-d1f5-11eb-9e18-88fc550b1e92.png)

Mutual Information Metric Graph

![image](https://user-images.githubusercontent.com/38917811/122679936-a0ce4f00-d1f5-11eb-8571-43e35c8ccd35.png)

## Optimization

 + The basic input to an optimizer is a cost function or metric object.
 
 + The metric is set using optimization algorithm. 

 + Once the optimization has finished, the final parameters can be obtained. 
 + Gradient Descent method was used in the Project.

## Transformation

 + This stage is where the conversion is done between the two images.

 + We performed your process using the Euler 3D method, which is more recommended for geometric transformation in the literature.

![image](https://user-images.githubusercontent.com/38917811/122680000-e3902700-d1f5-11eb-9a6b-6e3f54e97d32.png)

## Registration Result

  We have a poor result in terms of geometrically smooth image quality in our tomography image. In MR, we have a very good geometric output in terms of image quality and a poor output in terms of size. So how are things in our fusion image? Let's focus especially on the bone area. Geometrically it is the same as in the tomography view, and in terms of image quality it is the same as in the MRI image. So an image that meets our expectations.

![image](https://user-images.githubusercontent.com/38917811/122680009-f276d980-d1f5-11eb-83b8-7a33809c515d.png)

## Alpha Blending

It combines the pixels in the two images and allows the images we have joined to be seen overlapping.


PURPLE : Purple Areas are images obtained as a result of tomography. <br/>
GREEN  : Green Areas are images obtained as a result of MRI.

![image](https://user-images.githubusercontent.com/38917811/122680081-51d4e980-d1f6-11eb-9b5d-0c369f58ef72.png)

## Landmark Validation

![image](https://user-images.githubusercontent.com/38917811/122680100-5bf6e800-d1f6-11eb-8b37-dc5fd0da9a9f.png)

![image](https://user-images.githubusercontent.com/38917811/122680106-60bb9c00-d1f6-11eb-8ab4-a6fc9610d2e7.png)

# DEEP LEARNING METHODS!

## Registration Without Any Feature Extraction ( Classic Deep Learning )

 + 15 patient data with jpeg format was read.
 + Output layer activation function is sigmoid applied.
 + Occuracy %30

![image](https://user-images.githubusercontent.com/38917811/122680179-ac6e4580-d1f6-11eb-8b27-35a32bb7af23.png)

## Canny Edge Feature Extraction

 + Sk-Image library used for feature extraction.
 + Suitable for framing the Picture.

![image](https://user-images.githubusercontent.com/38917811/122680221-d9baf380-d1f6-11eb-840e-455310653346.png)

## Sobel Edge Detection Feature Extraction
 + Sk-Image library used for feature extraction.
 + More advanced version of the high-pass filter.

![image](https://user-images.githubusercontent.com/38917811/122680279-0707a180-d1f7-11eb-872f-b6311c28480a.png)

## AUTOENCODERS

 + The purpose of the autoencoder is to learn a symbolic vector that represents the data.

![image](https://user-images.githubusercontent.com/38917811/122680291-138bfa00-d1f7-11eb-87b2-e5738876b83e.png)

## ARRANGEMENT THE NEURAL NETWORK’S HIDDEN LAYERS
 + Output activation function changed as ReLU.
 + More hidden layer added.
 + Accuracy %94

![image](https://user-images.githubusercontent.com/38917811/122680328-42a26b80-d1f7-11eb-9fcf-638632459cd5.png)

## HIGH PASS & LOW PASS FEATURE EXTRACTION METHOD!

High-Pass : It was not used because of Sobel Edge Detection.
Low-Pass  : It was used to help improve accuracy in MR and CT images. The result of the study was yield.

## Superpixel Segmentation Benchmark Work

![image](https://user-images.githubusercontent.com/38917811/122680371-7382a080-d1f7-11eb-8e07-c26b6f5713a2.png)  

![image](https://user-images.githubusercontent.com/38917811/122680378-77aebe00-d1f7-11eb-9546-6b074a12c105.png)

![image](https://user-images.githubusercontent.com/38917811/122680382-7bdadb80-d1f7-11eb-818f-44615aa6fa1e.png)

![image](https://user-images.githubusercontent.com/38917811/122680386-7ed5cc00-d1f7-11eb-8bb4-3eab0e34edce.png)

# Result
 + Mean absolute error decrease to 2.72.
 + Accuracy increased by 3.7 after superpixel and low-pass filter feature extractions.
 + Accuracy %97.7
 
 ![image](https://user-images.githubusercontent.com/38917811/122680408-a2991200-d1f7-11eb-8d4d-e6b03f53b19a.png)














 
 
  
  
  
  
  
  
  
  
  
 













