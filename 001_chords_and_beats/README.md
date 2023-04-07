Plays piano chords for a song script copy pasted from guitar-tabs. See the [hello.txt](./hello.txt) file for an example of the format.

```
usage: player.py [-h] -f FILE [-o OCTAVE] [-b BEAT_DURATION] [-d DRUMS]
                 [-ab AFTER_BEATS]
```

Example usage:
Play the piano for Adele's hello (courtesy guitar-tab) and add drum beats (simple hi-hats) after 4 4-beat cycles (i.e. 16 beats).

```
 python player.py -f hello.txt -d 42 -ab 16
```