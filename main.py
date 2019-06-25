import googlemaps
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import urllib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
from skimage import data, io, filters, feature, measure, morphology, segmentation, color
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.morphology import watershed
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans


addr = "Raigad Fort Natural Reserve, Maharashtra"

def getgeocodfromadd(add):
    
    gmaps = googlemaps.Client(key="AIzaSyCvwb-MpxoyGLro17bWbkxfctLhfop_reI")
    geocode_result = gmaps.geocode(add)
    #print(geocode_result)
    res = geocode_result[0]['geometry']['location']
    return res

def getpic(loc):
    lat = str(loc["lat"])
    lon = str(loc["lng"])
    key = "AIzaSyCbp4l7KZp0KVZ86lFzzIFcGH9rPObvj4E"
    url = "http://maps.googleapis.com/maps/api/staticmap?center="
    #path = 
    url1 = "&size=1200x1200&maptype=satellite&zoom=15&key="+key
    url2 = url + lat + "," + lon + url1
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
                      'AppleWebKit/537.11 (KHTML, like Gecko) '
                      'Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    request = urllib.request.Request(url2, None, hdr)#{'User-Agent': user_agent}) 
    infile = urllib.request.urlopen(request)
    data = infile.read()
    buffer2 = StringIO(data)
    image2 = Image.open(buffer2)
    
    return image2

def for_cvr(masks):
    x = masks.shape
    ct = 0
    for i in range(0,x[0]):
        for j in range(0,x[1]):
            if masks[i][j]==1:
                ct+=1
        
    fr_cvg = ct/masks.size
    fr_cvg = fr_cvg*100

    return fr_cvg

def compute(addr):
    loc = getgeocodfromadd(addr)
    image = getpic(loc)

    #image = io.imread('staticmap3.png')

    plt.imshow(image)

    img2 = image.copy()
    mask = image[:,:,0] > 80
    img2[mask] = [0, 0, 0]

    mask = image[:,:,2] > 80
    img2[mask] = [0, 0, 0]
    #plt.imshow(img2)

    img3 = rgb2gray(img2)
    plt.imshow(img3, 'gray')
    plt.hist(img3.ravel(), bins=256, histtype='step', color='black');
    elevation = sobel(img3)
    plt.imshow(elevation)
    img3_denoised = filters.median(img3, selem=np.ones((7,7)))

    f, (ax0, ax1) = plt.subplots(1,2,figsize=(15,5))
    ax0.imshow(img3)
    ax1.imshow(img3_denoised)

    masks = np.zeros_like(img3_denoised)
    masks[img3_denoised>0.2] = 1
    plt.imshow(masks)

    edges = feature.canny(img3_denoised, sigma=3)

    dt = distance_transform_edt(~edges)
    plt.imshow(dt)

    local_max = feature.peak_local_max(dt, indices=False, min_distance=5)
    plt.imshow(local_max, cmap='gray');

    peak_idx = feature.peak_local_max(dt, indices=False, min_distance=5)
    peak_idx[:5]

    plt.plot(peak_idx[:,1], peak_idx[:,0], 'r.')
    plt.imshow(dt)

    markers = measure.label(local_max)
    labels = morphology.watershed(-dt, markers)
    plt.imshow(segmentation.mark_boundaries(image, labels));
    plt.imshow(color.label2rgb(labels, image));
    plt.imshow(color.label2rgb(labels, image, kind='avg'));
    regions = measure.regionprops(labels, intensity_image=img3)
    region_means = [r.mean_intensity for r in regions]
    plt.hist(region_means, bins=20);

    model = KMeans(n_clusters=2)
    region_means = np.array(region_means).reshape(-1,1)

    model.fit(region_means)
    print(model.cluster_centers_)

    bg_fg_labels = model.predict(region_means)
    #bg_fg_labels

    classified_labels = labels.copy()
    for bg_fg, region in zip(bg_fg_labels, regions):
        classified_labels[tuple(region.coords.T)] = bg_fg

    plt.imshow(color.label2rgb(classified_labels, image));

    fr_cvg = for_cvr(masks)
    print(fr_cvg)

def main():
    f = open('regions2.txt', 'r')
    for data in f:
        compute(data)

main()
