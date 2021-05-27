import numpy as np
import pymeanshift as pms


def sliceImage(imgArr: np.array, newSize: tuple) -> dict:
    '''
    Creates slices of an image.

    Arguments:
        - imgArr  : image array used for slicing
        - newSize : width and height of new slices
    
    returns:  dictionary containing the following information
    {
        "slicedImages": [ (xStartPos, yStartPos, slicedImgArr) ],
        "sliceCount"  : int
    }
    OR empty dictionary if slicing was unable to be completed
    '''
    width     = imgArr.shape[0]
    height    = imgArr.shape[1]
    newWidth  = newSize[0]
    newHeight = newSize[1]

    # Return empty dictionary if desired slice is larger than image
    if newWidth > width or newHeight > height:
        return {}

    # Return data
    newSliceCount = 0
    imageList     = []

    # Process image
    for i in range(0, width, newWidth):
        for j in range(0, height, newHeight):
            newSliceCount += 1
            
            # Create array of zeros of slice size
            base = np.zeros((newWidth, newHeight, imgArr.shape[2]))

            # Slice image and broadcase to base array
            # Slice will be blacked out when outside range of original image
            sliceArr = imgArr[i:i+newWidth, j:j+newHeight, :]
            base[ :sliceArr.shape[0], :sliceArr.shape[1]] = sliceArr
            imageList.append((i, j, base))

    return {
        "slicedImages": imageList,
        "sliceCount"  : newSliceCount
    }


def meanShiftSegmentation(imgArr: np.array, spatialRadius: int=6, rangeRadius: int=6, minDensity: int=50) -> tuple:
    '''
    SOURCE - https://github.com/fjean/pymeanshift

    Arguments:
        - imgArr       : image array to use for algo
        - spatialRadius: spatial radius for ms algo
        - rangeRadius  : range radius for ms algo
        - minDensity   : min density for ms algo
    
    return: (segmented_image, labels_image, number_regions) 
    '''
    return pms.segment(imgArr, 
                       spatial_radius=spatialRadius, 
                       range_radius=rangeRadius, 
                       min_density=minDensity)


# TO DO::Fix comments
def sliceAndMeanShift(imgArr: np.array, size: int):
    '''
    Returns dictionary:
    {
        "slices":
        {
            "128": [ (x, y, imgArr) ], <- x, y are pixel start positions in original image
            "256": ...,
            "512":
        },

        "meanShift":
        {
            "128": [ (x, y, numRegions, segImage, labelImage) ], 
            "256": ...,
            "512":
        }
    }
    '''
    finalResults = {
        "slices":
        {
            "128": [],
            "256": [],
            "512": []
        },

        "meanShift":
        {
            "128": [], 
            "256": [],
            "512": []
        }
    }
    
    if imgArr.shape[0] >= size and imgArr.shape[1] >= size:
        # Step 4: Slice 
        sliceResults = sliceImage(imgArr, (size, size))
         # Step 5: Save slices
        finalResults["slices"][str(size)] = sliceResults["slicedImages"] 
        # Step 6: Mean Shift on all slices
        for res in sliceResults["slicedImages"]:
            i = res[0]
            j = res[1]
            slicedImg = res[2].astype(np.uint8)
            # mean shift            
            (segmentedImage, labelsImage, numberRegions) = meanShiftSegmentation(slicedImg)
            # add to results
            finalResults["meanShift"][str(size)].append((i, j, numberRegions, segmentedImage, labelsImage))

    return finalResults