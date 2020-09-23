from django.urls import path
from . import views

app_name = 'app'
urlpatterns = [
    path('', views.movie_index),
    path('movies/', views.movie_index),
    path('page/<int:id>/', views.movies),
    path('movies/page/<int:id>/', views.movies),
    path('actors/', views.actor_index),
    path('actors/page/<int:id>/', views.actors),
    path('movie/<int:id>/', views.movie),
    path('actor/<int:id>/', views.actor),
    path('movies/search/', views.movie_search),
    path('movies/search/page/<int:id>', views.movie_search_page),
    path('actors/search/', views.actor_search),
    path('actors/search/page/<int:id>', views.actor_search_page),
    path('reviews/search/', views.review_search),
    path('reviews/search/page/<int:id>', views.review_search_page),
]
