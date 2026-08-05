from os import path
from logging import Logger
from optparse import OptionParser

from beets import dbcore
from beets.library import Library, Item, parse_query_string
from beets.ui import Subcommand, decargs
from beetsplug.essentia import EssentiaInterface
from confuse import ConfigView

plg_ns = {}
about_path = path.join(path.dirname(__file__), u'about.py')
with open(about_path) as about_file:
    exec(about_file.read(), plg_ns)

class EssentiaCommand(Subcommand):
    config: ConfigView = None

    lib = None
    query = None
    parser = None

    items_to_analyse = None

    def __init__(self, config: ConfigView, log: Logger):
        self.config = config
        self.log = log

        cfg_dry_run = self.config["dry-run"].get()
        cfg_write = self.config["write"].get()
        cfg_threads = self.config["threads"].get()
        cfg_force = self.config["force"].get()
        cfg_version = False
        cfg_count_only = False
        cfg_quiet = self.config["quiet"].get()

        self.parser = OptionParser(usage='beet {plg} [options] [QUERY...]'.format(plg=plg_ns['__PLUGIN_NAME__']))

        self.parser.add_option(
            '-d', '--dry-run',
            action='store_true', dest='dryrun', default=cfg_dry_run,
            help=u'[default: {}] only show what would be done without '
                 u'modifying library items or files'.format(cfg_dry_run)
        )

        self.parser.add_option(
            '-w', '--write',
            action='store_true', dest='write', default=cfg_write,
            help=u'[default: {}] write the extracted values (bpm, mood) to the '
                 u'media files'.format(cfg_write)
        )

        self.parser.add_option(
            '-t', '--threads',
            action='store', dest='threads', type='int', default=cfg_threads,
            help=u'[default: {}] the number of threads to run in parallel'.format(cfg_threads)
        )

        self.parser.add_option(
            '-f', '--force',
            action='store_true', dest='force', default=cfg_force,
            help=u'[default: {}] force analysis of items that already have '
                 u'values for enabled tags'.format(cfg_force)
        )

        self.parser.add_option(
            '-c', '--count-only',
            action='store_true', dest='count_only', default=cfg_count_only,
            help=u'[default: {}] Show the number of items to be processed'.format(cfg_count_only)
        )

        self.parser.add_option(
            '-q', '--quiet',
            action='store_true', dest='quiet', default=cfg_quiet,
            help=u'[default: {}] mute all output'.format(cfg_quiet)
        )

        self.parser.add_option(
            '-v', '--version',
            action='store_true', dest='version', default=cfg_version,
            help=u'show plugin version'
        )

        # Keep this at the end
        super(EssentiaCommand, self).__init__(
            parser=self.parser,
            name=plg_ns['__PLUGIN_NAME__'],
            aliases=[plg_ns['__PLUGIN_ALIAS__']] if
            plg_ns['__PLUGIN_ALIAS__'] else [],
            help=plg_ns['__PLUGIN_SHORT_DESCRIPTION__']
        )

    def func(self, lib: Library, options, arguments):
        self.config['dry-run'].set(options.dryrun)
        self.config['write'].set(options.write)
        self.config['force'].set(options.force)
        self.config['quiet'].set(options.quiet)
        self.config['threads'].set(options.threads)

        self.lib = lib
        self.query = decargs(arguments)

        if options.version:
            self.show_version_information()
            return

        self.analyse(options.count_only, options.force)

    def analyse(self, count_only: bool, force: bool):
        self.find_items_to_analyse(force)
        self.log.info("Number of items to be analysed: {}".format(len(self.items_to_analyse)), False)

        # Count only and exit
        if count_only:
            return

        # Run tasks on selected items
        es = EssentiaInterface(self.config, self.log)
        es.analyse(self.items_to_analyse)

    def find_items_to_analyse(self, force: bool):
        # Parse the incoming query
        parsed_query, parsed_sort = parse_query_string(" ".join(self.query), Item)
        combined_query = parsed_query

        # Add unprocessed items query
        if not force:
            # Set up the query for unprocessed items
            subqueries = []
            tags = self.config['tags']
            for fld in tags.keys():
                if tags[fld]["enabled"].get(bool):
                    fast = fld in Item._fields
                    query_item = dbcore.query.MatchQuery(fld, None, fast=fast)
                    subqueries.append(query_item)

            unprocessed_items_query = dbcore.query.OrQuery(subqueries)
            combined_query = dbcore.query.AndQuery([parsed_query, unprocessed_items_query])

        self.log.debug("Combined query: {}".format(combined_query))

        # Get the library items
        self.items_to_analyse = self.lib.items(combined_query, parsed_sort)
        if len(self.items_to_analyse) == 0:
            self.log.info("No items to process")
            return

    def show_version_information(self):
        self.log.info("{pt}({pn}) plugin for Beets: v{ver}".format(
            pt=plg_ns['__PACKAGE_TITLE__'],
            pn=plg_ns['__PACKAGE_NAME__'],
            ver=plg_ns['__version__']
        ))
