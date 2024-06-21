from django.shortcuts import render
import joblib
from django.contrib import messages
import pandas as pd
from sklearn.model_selection import train_test_split
def home(request):
        df = pd.read_excel('../final.xlsx')
        df.dropna()
        X = df.drop(columns=['id', 'diagnosis'], axis=1)
        Y = df['diagnosis']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        loadJobLib = joblib.load("final.pkl")
        testing = loadJobLib.predict(X_test)

        return render(request, 'index.html', {"prediction": testing})

def results(request):
        if request.method == 'POST':
                radius_mean = float(request.POST.get('radius_mean'))
                texture_mean = float(request.POST.get('texture_mean'))
                perimeter_mean = float(request.POST.get('perimeter_mean'))
                area_mean = float(request.POST.get('area_mean'))
                smoothness_mean = float(request.POST.get('smoothness_mean'))
                compactness_mean = float(request.POST.get('compactness_mean'))
                concavity_mean = float(request.POST.get('concavity_mean'))
                concave_points_mean = float(request.POST.get('concave_points_mean'))
                symmetry_mean = float(request.POST.get('symmetry_mean'))
                fractal_dimension_mean = float(request.POST.get('fractal_dimension_mean'))
                radius_se = float(request.POST.get('radius_se'))
                texture_se = float(request.POST.get('texture_se'))
                perimeter_se = float(request.POST.get('perimeter_se'))
                area_se = float(request.POST.get('area_se'))
                smoothness_se = float(request.POST.get('smoothness_se'))
                compactness_se = float(request.POST.get('compactness_se'))
                concavity_se = float(request.POST.get('concavity_se'))
                concave_points_se = float(request.POST.get('concave_points_se'))
                symmetry_se = float(request.POST.get('symmetry_se'))
                fractal_dimension_se = float(request.POST.get('fractal_dimension_se'))
                radius_worst = float(request.POST.get('radius_worst'))
                texture_worst = float(request.POST.get('texture_worst'))
                perimeter_worst = float(request.POST.get('perimeter_worst'))
                area_worst = float(request.POST.get('area_worst'))
                smoothness_worst = float(request.POST.get('smoothness_worst'))
                compactness_worst = float(request.POST.get('compactness_worst'))
                concavity_worst = float(request.POST.get('concavity_worst'))
                concave_points_worst = float(request.POST.get('concave_points_worst'))
                symmetry_worst = float(request.POST.get('symmetry_worst'))
                fractal_dimension_worst = float(request.POST.get('fractal_dimension_worst'))

                data = [radius_mean,texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
                        symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                        concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
                loadjobLib = joblib.load("final.pkl")

                answer = loadjobLib.predict([data])
                messages.success(request, answer[0])
        return render(request, 'prediction.html')