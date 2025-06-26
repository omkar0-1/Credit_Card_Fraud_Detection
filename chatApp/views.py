from django.views.generic import TemplateView


class HomePage(TemplateView):
    template_name = 'index.html'


def info(request):
    return render(request, 'info.html')


def DETECTION_PAGE(request):
    return render(request, 'detection.html')


from django.shortcuts import render
import numpy as np
import joblib

model = joblib.load("fraud_detection_model.sav")


def classifier(request):
    global features
    if request.method == "POST":
        # Get the form data and convert to a NumPy array
        feature1 = request.POST.get('feature1')
        feature2 = request.POST.get('feature1')
        feature3 = request.POST.get('feature1')
        feature4 = request.POST.get('feature1')
        feature5 = request.POST.get('feature1')
        feature6 = request.POST.get('feature1')
        feature7 = request.POST.get('feature1')
        feature8 = request.POST.get('feature1')
        feature9 = request.POST.get('feature1')
        feature10 = request.POST.get('feature1')
        feature11 = request.POST.get('feature1')
        feature12 = request.POST.get('feature1')
        feature13 = request.POST.get('feature1')
        feature14 = request.POST.get('feature1')
        feature15 = request.POST.get('feature1')
        feature16 = request.POST.get('feature1')
        feature17 = request.POST.get('feature1')
        feature18 = request.POST.get('feature1')
        feature19 = request.POST.get('feature1')
        feature20 = request.POST.get('feature1')
        feature21 = request.POST.get('feature1')
        feature22 = request.POST.get('feature1')
        feature23 = request.POST.get('feature1')
        feature24 = request.POST.get('feature1')
        feature25 = request.POST.get('feature1')
        feature26 = request.POST.get('feature1')
        feature27 = request.POST.get('feature1')
        feature28 = request.POST.get('feature1')
        feature29 = request.POST.get('feature1')

        # Add more features here as needed...

        # Create a feature array
        features = np.array(
            [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
             feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19,
             feature20, feature21, feature22, feature23, feature24, feature25, feature26, feature27, feature28,
             feature29]).reshape(1, -1)

        # Make a prediction using the model
        prediction = model.predict(features)

        if prediction == 1:
            return render(request, "fraudulent.html")
        else:
            return render(request, "not_fraudulent.html")
    else:
        return render(request, "classifier.html")


def predict(request):
    return render(request, 'classifier.py')
