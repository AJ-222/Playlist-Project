import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

clientID = "0ca8171b113f42728f64fb5e3f0fa4cc"
clientSecret = "61c6e3fdef144bc6abfafbdd7dd08701"

authManager = SpotifyClientCredentials(client_id=clientID, client_secret= clientSecret)
sp = spotipy.Spotify(auth_manager=authManager)
