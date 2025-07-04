"""chatApp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import to include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

from accounts import urls as accounts_urls
from . import views
from . import views as base_views

urlpatterns = [
    path('', base_views.HomePage.as_view(), name='home'),
    path('accounts/', include(accounts_urls)),
    path('detection/', views.DETECTION_PAGE, name='detection'),
    path('info/', views.info, name='info'),
    path('classifier/predict/', views.classifier, name='predict'),
    path('classifier/', views.classifier, name='classifier')

]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)
