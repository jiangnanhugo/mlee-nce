#run the program during the spare time either weekends or between 9:00 PM and 5:00 AM Eastern time during weekdays

use LWP::Simple;
use Encode;;

#for ($year=1940;$year<=2011;$year++)
for ($year=2012;$year>=2012;$year--)
{
	#$query = 'science[journal]+AND+breast+cancer+AND+2008[pdat]';
	$s_year ="$year";
	$query = $s_year."[pdat]";
	unless(mkdir $s_year) {
		die "Unable to create $s_year\n";
	}
	
	#assemble the esearch URL
	$base = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/';
	#$url = $base . "esearch.fcgi?db=pubmed&tool=load_med_nlp&email=zhoudeyu@gmail.com&term=$query&usehistory=y";
	$url = $base."esearch.fcgi?db=pubmed&term=$query&usehistory=y";

	#post the esearch URL
	$output = get($url);
	
	#parse WebEnv, QueryKey and Count (# records retrieved)
	$web = $1 if ($output =~ /<WebEnv>(\S+)<\/WebEnv>/);
	$key = $1 if ($output =~ /<QueryKey>(\d+)<\/QueryKey>/);
	$count = $1 if ($output =~ /<Count>(\d+)<\/Count>/);
	
	print "$web $key $count"
	
	#open output file for writing
	
	
	# #retrieve data in batches of 500
	# $retmax = 100;
	# for ($retstart = 0; $retstart < $count; $retstart += $retmax) 
	# {
	# 		open OUT, ">:utf8", "$year/".$retstart|| die "Can't open file!\n";
	# 		#open(OUT, ">$year/".$retstart) || die "Can't open file!\n";
	#         $efetch_url = $base ."efetch.fcgi?db=pubmed&WebEnv=$web";
	#         $efetch_url .= "&query_key=$key&retstart=$retstart";
	#         #$efetch_url .= "&retmax=$retmax&rettype=abstract&retmode=text";
	#         $efetch_url .= "&retmax=$retmax&retmode=xml";

	#         $efetch_out = get($efetch_url);
	#         $efetch_out = Encode::decode_utf8($efetch_out);
	#         if ( defined $efetch_out)
	#         {
	#         	print OUT "$efetch_out";
	#         }	
	# }
	# close OUT;
}	
