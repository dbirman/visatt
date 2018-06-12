# visatt

conda create -n py3att python=3.6.5 anaconda

source activate py3att

# useful reading

2017 model to do eye fixation prediction, basically VGG with some weird bias layers on top
Kruthiventi, S. S., Ayush, K., & Babu, R. V. (2017). Deepfix: A fully convolutional neural network for predicting human eye fixations. IEEE Transactions on Image Processing, 26(9), 4446-4456.

Early 2015 model to do eye fixation prediction, fixation-centered with variable image sizes, much closer to what you want to do
Liu, N., Han, J., Zhang, D., Wen, S., & Liu, T. (2015, June). Predicting eye fixations using convolutional neural networks. In Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on (pp. 362-370). IEEE.

# initial data

Download from: http://www.inb.uni-luebeck.de/fileadmin/files/MITARBEITER/Dorr/EyeMovementDataSet.html

Video frames with eye tracking

# model spec

Take an original video and eye-tracking data
For each frame crop crop the video into 128x128, 256x256, and 512x512 pixel views, compressed to 64x64. 
Each view gets passed through a conv/relu/pool layer four times (64x64->32x32->16x16->8x8) with K kernels at each layer (16 maybe?) and all convolutions at 3x3
The final layer 8x8x16x3 gets concatenated to 8x8x48 and deconvolved to a 64x64 map in two steps (8x8x48->16x16x16->64x64x1)
Final output is logistic not relu
The output is compared to a gaussian-blurred map of the eye position in the window +175ms to +225 ms (center can be shifted but by default 200 ms)

Cross-entropy loss

Pre-train convolutional layers on imagenet with FC layers on the back end, this approximates alexnet but one very small images. 
Train a base model first on free-viewing data
Re-train model for active viewing (e.g. search for people, etc)


# datasets

Each dataset is pulled and stored in 

Pulled from Mnih, V., Heess, N., & Graves, A. (2014). Recurrent models of visual attention. In Advances in neural information processing systems (pp. 2204-2212).

24. T. Judd, K. Ehinger, F. Durand, and A. Torralba, “Learning to predict where humans look,” ICCV
2009.
25. N. Bruce, J. Tsotsos, “Attention based on information maximization,” Journal of Vision, 2007.
26. S. Ramanathan et al., “An eye fixation database for saliency detection in images,” ECCV 2010.
27. K. Ehinger, et al., “Modeling search for people in 900 scenes: A combined source model of eye
guidance,” Visual Cognition, vol. 17, pp. 945-978, 2009.
28. H. Hadizadeh, M. J. Enriquez, and I. V. Bajić, “Eye-tracking database for a set of standard video
sequences,” IEEE Trans. Image Processing, vol. 21, no. 2, pp. 898-903, Feb. 2012.
29. S. Mathe and C. Sminchisescu, “Dynamic eye movement datasets and learnt saliency models for
visual action recognition, ECCV 2012.
30. L. Itti, R. Carmi, “Eye-tracking data from human volunteers watching complex video stimuli,” 2009
CRCNS.org.
31. T. Judd, F. Durand, and A. Torralba, “A benchmark of computational models of saliency to predict
human fixations,” Computer Science and Artificial Intelligence Laboratory Technical Report, 2012.
32. S. Winkler and R. Subramanian, “Overview of eye tracking datasets,” 2013 Fifth International
Workshop on Quality of Multimedia Experience (QoMEX), July 2013, Austria.
33. T. Judd, F. Durand, and A. Torralba: “Fixations on low-resolution images,” J. Vision, vol. 11, no. 4,
April 2011, http://people.csail.mit.edu/tjudd/SaliencyBenchmark/
34. J. Li et al., “Visual saliency based on scale-space analysis in the frequency domain.” IEEE Trans.
PAMI, vol. 35, no. 4, pp. 996–1010, April 2013.
35. M. Cerf, J. Harel, W. Einhäuser, C. Koch, “Predicting human gaze using low-level saliency combined
with face detection,” in Proc. NIPS, vol. 20, pp. 241–248, Vancouver, Canada, Dec. 3–8, 2007.
36. M. Dorr, T. Martinetz, K. Gegenfurtner, E. Barth, “Variability of eye movements when viewing
dynamic natural scenes,” J. Vision, vol. 10, no. 10, 2010. 
36
37. J. Wang, D. M. Chandler, P. Le Callet, “Quantifying the relationship between visual salience and
visual importance,” in Proc. SPIE, vol. 7527, San Jose, CA, Jan. 17–21, 2010.
38. G. Kootstra, B. de Boer, L. R. B. Schomaker, “Predicting eye fixations on complex visual stimuli
using local symmetry,” Cognitive Computation, vol. 3, no. 1, pp. 223–240, 2011.
39. I. van der Linde, U. Rajashekar, A. C. Bovik, L. K. Cormack, “DOVES: A database of visual eye
movements,” Spat. Vis., vol. 22, no. 2, pp. 161–177, 2009.
40. N. D. B. Bruce, J. K. Tsotsos, “Saliency based on information maximization.” in Proc. NIPS, vol. 19,
pp. 155–162, Vancouver, Canada, Dec. 4–9, 2006.
41. H. Liu, I. Heynderickx, “Studying the added value of visual attention in objective image quality
metrics based on eye movement data,” in Proc. ICIP, Cairo, Egypt, Nov. 7–10, 2009.
42. H. Alers, L. Bos, I. Heynderickx, “How the task of evaluating image quality influences viewing
behavior,” in Proc. QoMEX, Mechelen, Belgium, Sept. 7–9, 2011.
43. J. Redi, H. Liu, R. Zunino, I. Heynderickx: “Interactions of visual attention and quality perception.”
in Proc. SPIE, vol. 7865, San Jose, CA, Jan. 24–27, 2011.
44. U. Engelke, A. J. Maeder, H.-J. Zepernick, “Visual attention modeling for subjective image quality
databases,” in Proc. MMSP, Rio de Janeiro, Brazil, Oct. 5–7, 2009.
45. P. K. Mital, T. J. Smith, R. L. Hill, J. M. Henderson, “Clustering of gaze during dynamic scene
viewing is predicted by motion,” Cognitive Computation, vol. 3, no. 1, pp. 5–24, March 2011.
46. S. Péchard, R. Pépion, P. Le Callet, “Region-of-interest intra prediction for H.264/AVC error
resilience,” in Proc. ICIP, Cairo, Egypt, Nov. 7–10, 2009.
47. U. Engelke, R. Pepion, P. Le Callet, H.-J. Zepernick, “Linking distortion perception and visual
saliency in H.264/AVC coded video containing packet loss,” in Proc. VCIP, Huang Shan, China, July
11–14, 2010.
48. H. Alers, J. A. Redi, I. Heynderickx, “Examining the effect of task on viewing behavior in videos
using saliency maps,” in Proc. SPIE, vol. 8291, San Jose, CA, Jan. 22–26, 2012.
49. L. Itti, “Automatic foveation for video compression using a neurobiological model of visual
attention,” IEEE Trans. Image Processing, vol. 13, no. 10, pp. 1304–1318, Oct. 2004.
50. Z. Li, S. Qin, L. Itti, “Visual attention guided bit allocation in video compression,” Image Vis. Comp.,
vol. 29, no. 1, pp. 1–14, 2011. 
