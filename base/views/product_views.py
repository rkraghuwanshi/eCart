from typing_extensions import final
from django.shortcuts import render

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from base.models import Product, Review
from base.serializers import ProductSerializer

from rest_framework import status

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import joblib
import numpy as np
import pandas as pd


@api_view(['GET'])
def getProducts(request):
    query = request.query_params.get('keyword')
    if query == None:
        query = ''

    products = Product.objects.filter(
        name__icontains=query).order_by('-createdAt')
    print("------------------------------------")
    print(products)
    page = request.query_params.get('page')
    paginator = Paginator(products, 5)

    try:
        products = paginator.page(page)
    except PageNotAnInteger:
        products = paginator.page(1)
    except EmptyPage:
        products = paginator.page(paginator.num_pages)

    if page == None:
        page = 1

    page = int(page)
    print('Page:', page)
    serializer = ProductSerializer(products, many=True)
    # print(serializer.data)
    return Response({'products': serializer.data, 'page': page, 'pages': paginator.num_pages})


@api_view(['GET'])
def getTopProducts(request):
    products = Product.objects.filter(rating__gte=4).order_by('-rating')[0:5]
    serializer = ProductSerializer(products, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def getProduct(request, pk):
    product = Product.objects.get(_id=pk)

    serializer = ProductSerializer(product, many=False)

    description_lst = serializer.data['description'].split(' ')

    product_descriptions = pd.read_csv(
        'C:/Users/ASUS/Downloads/product_descriptions.csv')
    product_descriptions = product_descriptions.dropna()
    product_descriptions1 = product_descriptions.head(500)
    # product_descriptions1.iloc[:,1]

    product_descriptions1["product_description"].head(10)

    model = joblib.load('final_model.sav')
    vectorizer = TfidfVectorizer(stop_words='english')
    X1 = vectorizer.fit_transform(product_descriptions1["product_description"])

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    cluster_lst = []
    for item in description_lst:
        if(len(item) > 3):
            Y = vectorizer.transform([item])
            prediction = model.predict(Y)
            cluster_lst.append(prediction[0])
            print(prediction[0])
    if(len(cluster_lst) > 0):
        print_cluster(max(cluster_lst, key=cluster_lst.count),
                      order_centroids, terms)

        product_cluster = max(cluster_lst, key=cluster_lst.count)
        print("product_cluster = ", product_cluster)
        # Y = vectorizer.transform([description_lst[12]])
        # prediction = model.predict(Y)
        # print(prediction)
        # print_cluster(prediction[0], order_centroids, terms)
        print(serializer.data)
        # print(description_lst)
        query = ''
        result = products = Product.objects.filter(
            name__icontains=query)

        result_serialized = ProductSerializer(result, many=True)

        name_lst = []
        print("==========================================================================================")
        print(len(result_serialized.data))

        product_lst = []

        for i in range(len(result_serialized.data)):
            name_lst = result_serialized.data[i]['description'].split(" ")
            cluster_lst1 = []
            max_cluster = -1
            for item in name_lst:
                if(len(item) > 3):
                    Y = vectorizer.transform([item])
                    prediction = model.predict(Y)
                    cluster_lst1.append(prediction[0])
            if(len(cluster_lst1) != 0):
                max_cluster = max(cluster_lst1, key=cluster_lst1.count)
            if max_cluster == product_cluster:
                print(max_cluster)
                print("$")
                product_lst.append(result_serialized.data.pop(i))

        res = {}
        res['recommend'] = product_lst
        res.update(serializer.data)
        print(res)
    # print(serializer.data['recommend'])
        return Response(res)
    return Response(serializer.data)


def print_cluster(i, order_centroids, terms):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


@api_view(['POST'])
@permission_classes([IsAdminUser])
def createProduct(request):
    user = request.user

    product = Product.objects.create(
        user=user,
        name='Sample Name',
        price=0,
        brand='Sample Brand',
        countInStock=0,
        category='Sample Category',
        description=''
    )

    serializer = ProductSerializer(product, many=False)
    return Response(serializer.data)


@api_view(['PUT'])
@permission_classes([IsAdminUser])
def updateProduct(request, pk):
    data = request.data
    product = Product.objects.get(_id=pk)

    product.name = data['name']
    product.price = data['price']
    product.brand = data['brand']
    product.countInStock = data['countInStock']
    product.category = data['category']
    product.description = data['description']

    product.save()

    serializer = ProductSerializer(product, many=False)
    return Response(serializer.data)


@api_view(['DELETE'])
@permission_classes([IsAdminUser])
def deleteProduct(request, pk):
    product = Product.objects.get(_id=pk)
    product.delete()
    return Response('Producted Deleted')


@api_view(['POST'])
def uploadImage(request):
    data = request.data

    product_id = data['product_id']
    product = Product.objects.get(_id=product_id)

    product.image = request.FILES.get('image')
    product.save()

    return Response('Image was uploaded')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def createProductReview(request, pk):
    user = request.user
    product = Product.objects.get(_id=pk)
    data = request.data

    # 1 - Review already exists
    alreadyExists = product.review_set.filter(user=user).exists()
    if alreadyExists:
        content = {'detail': 'Product already reviewed'}
        return Response(content, status=status.HTTP_400_BAD_REQUEST)

    # 2 - No Rating or 0
    elif data['rating'] == 0:
        content = {'detail': 'Please select a rating'}
        return Response(content, status=status.HTTP_400_BAD_REQUEST)

    # 3 - Create review
    else:
        review = Review.objects.create(
            user=user,
            product=product,
            name=user.first_name,
            rating=data['rating'],
            comment=data['comment'],
        )

        reviews = product.review_set.all()
        product.numReviews = len(reviews)

        total = 0
        for i in reviews:
            total += i.rating

        product.rating = total / len(reviews)
        product.save()

        return Response('Review Added')
