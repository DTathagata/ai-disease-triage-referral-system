from django.contrib import admin
from django.urls import path
from ml65app import views

urlpatterns = [
    path('admin/', admin.site.urls),   # Django admin panel
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('predict/', views.predict, name='predict'),
    path('contact/', views.contact, name='contact'),
]