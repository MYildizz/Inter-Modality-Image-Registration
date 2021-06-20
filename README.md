
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

# Interpolation Types

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

  
 
 
  
  
  
  
  
  
  
  
  
 













