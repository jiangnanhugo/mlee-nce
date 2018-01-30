################################################################################
# @Author Wei-Ming Chen, PhD                                                   #
# @pubmed abstracts scraper                                                    #
# @version: v1.0.0                                                             #
# @usage: perl abstract_scraper.pl                                             #
################################################################################

use utf8;
use LWP::Simple;
use threads;

# Maximum number of threads to be issued
my $maxThread = 24; 


# Download all PubMed abstracts
my $db = 'pubmed';
my $query = '"0000/01/01"[PDAT] : "3000/12/31"[PDAT]';
my $retmax = 1000;


# Assemble the esearch URL
my $base = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/';
my $url = $base . "esearch.fcgi?db=$db&term=$query&usehistory=y";
my $output = get($url);

open(my $fh, '>', 'summary.xml');
print $fh "$output";
close $fh;


# Parse WebEnv and QueryKey
my $web = $1 if ($output =~ /<WebEnv>(\S+)<\/WebEnv>/);
my $key = $1 if ($output =~ /<QueryKey>(\d+)<\/QueryKey>/);
my $count = $1 if ($output =~ /<Count>(\d+)<\/Count>/);


# Retrieve data in batches of 1000
for ($retstart = 0; $retstart < $count; $retstart += $retmax) {
	my $treadIssued = 'no';

	# loop until the thread is issued on this thread
	while ($treadIssued eq 'no') {
		# check num of running threads
		my @runningThrAry = threads->list(threads::running);
		
		# if running threads lower than the limit
		if (@runningThrAry < $maxThread) {
			# spawn a new thread
			threads->create(\&efetch, ($db, $key, $web, $retstart, $retmax));
			$treadIssued = 'yes';
			@runningThrAry = threads->list(threads::running);
		}
		sleep 1;
	}

	# detach the finished threads
	my @joinableThrAry = threads->list(threads::joinable);
	foreach my $joinableThr (@joinableThrAry) {
		$joinableThr->detach() if not $joinableThr->is_running();
	}
}
 

sub efetch {
	my ($db, $key, $web, $retstart, $retmax) = @_;
	my $efetch_url = $base . "efetch.fcgi?db=$db&query_key=$key&WebEnv=$web&retstart=$retstart&retmax=$retmax&retmode=xml";
	my $efetch_out = get($efetch_url);
	
	open(my $fh, '>', "$retstart".'.txt');
	print $fh "$efetch_out";
	close $fh;
	
	open(my $fh, '>>', 'esearch.err');
	print $fh "$retstart ";
	close $fh;
	
	print "$retstart ";
}