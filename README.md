[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

# beets-essentia Plugin

The _beets-essentia_ plugin lets you, through the use of the [Essentia](https://essentia.upf.edu/index.html) tensorflow models,
obtain low and high level musical information from your songs.

Currently, the following tags are extracted and applied for each library item:
`bpm`, `mood`

## Installation

The plugin can be installed via:

```shell script
pip install beets-essentia
```

and activated the usual way by adding `essentia` to the list of plugins in your configuration:

```yaml
plugins:
  - essentia
```

## Configuration

All your configuration will need to go under the `essentia` key. This is what the default configuration values look like:

```yaml
essentia:
  auto: no # auto execute on import
  dry-run: no # only display changes, but do not apply
  write: yes # write changes to the file metadata
  threads: auto # how many cpu threads to use
  force: no # force overwrite existing metadata entries (mainly for bpm)
  quiet: no # do not display logs
  path: auto # folder containing the tensorflow models (the models will be downloaded by default to ~/essentia)
  tags: # tagging settings
    bpm: # bpm
      enabled: yes # should this tag be analysed
      threshold: 0.1 # threshold for the models confidence (0.1 means that the model is 90% confident, that the bpm is right)
      # model to use 
      # -- this can also be an absolute path to a metadata file
      #    for choosing possible model metadata files look here: https://essentia.upf.edu/models.html
      # -- or this could be the path to the actual model and metadata files (without the .pb/.json ending)
      # if no file is present the model defined here will be automatically downloaded
      # from https://essentia.upf.edu/models/(model).(pb|json)
      model: "tempo/tempocnn/deeptemp-k16-3"
    mood: # mood
      enabled: yes
      mapping: mood # how the actual music tag should be named
      separator: ";" # what separator should be used to separate multiple moods
      force_overwrite: no # overwrite existing moods on force, instead of appending new moods
      moods: # the individual moods
        aggressive:
          enabled: yes
          threshold: 0.1
          mapping: # multiple names can be added to the mood list
            - aggressive
          model: "classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1"
        happy:
          enabled: yes
          threshold: 0.1
          mapping:
            - happy
          model: "classification-heads/mood_happy/mood_happy-discogs-effnet-1"
        party:
          enabled: yes
          threshold: 0.1
          mapping:
            - party
          model: "classification-heads/mood_party/mood_party-discogs-effnet-1"
        relaxed:
          enabled: yes
          threshold: 0.1
          mapping:
            - relaxed
          model: "classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1"
        sad:
          enabled: yes
          threshold: 0.1
          mapping:
            - sad
          model: "classification-heads/mood_sad/mood_sad-discogs-effnet-1"
        mirex:
          enabled: no
          threshold: 0.1
          model: "classification-heads/mood_mirex/moods_mirex-msd-musicnn-1"
          mapping:
            # the individual mirex categories have multiple adjacent moods
            # by default only one is activated per category
            category_1:
              - passionate
              #- rousing
              #- confident
              #- boisterous
              #- rowdy
            category_2:
              #- rollicking
              - cheerful
              #- fun
              #- sweet
              #- amiable
              #- good-natured
            category_3:
              #- literate
              #- poignant
              - wistful
              #- bittersweet
              #- autumnal
              #- brooding
            category_4:
              - humorous
              #- silly
              #- campy
              #- quirky
              #- whimsical
              #- witty
              #- wry
            category_5:
              - aggressive
              #- fiery
              #- tense
              #- anxious
              #- intense
              #- volatile
              #- visceral
        jamendo:
          enabled: no
          threshold: 0.1
          model: "classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs_track_embeddings-effnet-1"
          mapping:
            # the jamendo model handles not only moods, but also themes
            # by default only the moods are stored in the mood tag
            #action:
            #  - action
            #adventure:
            #  - adventure
            #advertising:
            #  - advertising
            #background:
            #  - background
            #ballad:
            #  - ballad
            calm:
              - calm
            #children:
            #  - children
            #christmas:
            #  - christmas
            #commercial:
            #  - commercial
            cool:
              - cool
            #corporate:
            #  - corporate
            dark:
              - dark
            deep:
              - deep
            #documentary:
            #  - documentary
            #drama:
            #  - drama
            dramatic:
              - dramatic
            #dream:
            #  - dream
            emotional:
              - emotional
            energetic:
              - energetic
            epic:
              - epic
            fast:
              - fast
            #film:
            #  - film
            #fun:
            #  - fun
            funny:
              - funny
            #game:
            #  - game
            groovy:
              - groovy
            happy:
              - happy
            heavy:
              - heavy
            #holiday:
            #  - holiday
            hopeful:
              - hopeful
            inspiring:
              - inspiring
            #love:
            #  - love
            meditative:
              - meditative
            melancholic:
              - melancholic
            melodic:
              - melodic
            motivational:
              - motivational
            #movie:
            #  - movie
            #nature:
            #  - nature
            party:
              - party
            positive:
              - positive
            powerful:
              - powerful
            relaxing:
              - relaxing
            #retro:
            #  - retro
            romantic:
              - romantic
            sad:
              - sad
            sexy:
              - sexy
            slow:
              - slow
            soft:
              - soft
            #soundscape:
            #  - soundscape
            #space:
            #  - space
            #sport:
            #  - sport
            #summer:
            #  - summer
            #trailer:
            #  - trailer
            #travel:
            #  - travel
            upbeat:
              - upbeat
            uplifting:
              - uplifting
```

## Usage

Invoke the plugin as:

    $ beet essentia [options] [QUERY...]

For a more verbose reporting use the `-v` flag on `beet`:

    $ beet -v essentia [options] [QUERY...]

The plugin has also got a shorthand `esnt` so you can also invoke it like this:

    $ beet esnt [options] [QUERY...]

The following command line options are available:

**--dry-run [-d]**: Only show what would be done - displays the extracted values but does not store them in the library.

**--write [-w]**: Write the values (bpm only) to the media files.

**--threads=THREADS [-t THREADS]**: The number of concurrently running executions.

**--force [-f]**: Force the analysis of all items (skip attribute checks).

**--count-only [-c]**: Show the number of items to be processed and exit. Extraction will not be executed.

**--quiet [-q]**: Run without any output.

**--version [-v]**: Display the version number of the plugin. Useful when you need to report some issue and you have to state the version of the plugin you are using.

These command line options will override those specified in the configuration file.

## Issues

- If something is not working as expected please use the Issue tracker.
- If the documentation is not clear please use the Issue tracker.
- If you have a feature request please use the Issue tracker.
- In any other situation please use the Issue tracker.

## Credits

Essentia is an open-source C++ library with Python bindings for audio analysis and audio-based music information retrieval. It is released under the Affero GPLv3 license and is also available under proprietary license upon request. This plugin is just a mere wrapper around this library. [Learn more about the Essentia project](http://essentia.upf.edu)

## References

- [Essentia](https://essentia.upf.edu/index.html)
- [Essentia Licensing](https://essentia.upf.edu/licensing_information.html)
