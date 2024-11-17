from django.urls import path
from . import views

urlpatterns = [
    path('', views.record_audio, name='record_audio'),
    # path('', views.process_audio, name='process_audio'),
    path('process/', views.process_audio, name='process_audio'),
    path('map/',views.maps_audio,name="maps")
]