<?php
	/**
	 *	GIT DEPLOYMENT SCRIPT
	 *
	 *	Pulls newest version of site from github repo @ github.com/empathetic-alligator/wpovell.net
	 *
	 *	Based off of https://gist.github.com/1809044
	 */

	// Commands to run
	$commands = array(
		'echo $PWD',
		'whoami',
		'git pull',
		'git status',
		'git rev-parse HEAD'
	);

	$f = fopen('/home/wpovelln/.secret_token', 'r');
	$SECRET_TOKEN = fgets($f);
	fclose($f);

	// Validate token
	if(!isset($_GET["pass"]) || $_GET["pass"] != $SECRET_TOKEN) {
		header("HTTP/1.1 403 Forbidden");
		$_GET['error'] = 403;
		include "error.php";
		exit;
	} else {
		$output = '';
		// Run the commands for output
		foreach($commands AS $command){
			// Run it
			$tmp = shell_exec($command);
			// Output
			$output .= "<span style=\"color: #6BE234;\">\$</span> <span style=\"color: #729FCF;\">{$command}\n</span>";
			$output .= htmlentities(trim($tmp)) . "\n";
		}
	}
?>
<!DOCTYPE HTML>
<html lang="en-US">
<head>
	<meta charset="UTF-8">
	<title>GIT DEPLOYMENT SCRIPT</title>
</head>
<body style="background-color: #000000; color: #FFFFFF; font-weight: bold; padding: 0 10px;">
<pre>
 .  ____  .    ____________________________
 |/      \|   |                            |
[| <span style="color: #FF0000;">&hearts;    &hearts;</span> |]  |      Git Deployment Script |
 |___==___|  /  Based on script by oodavid |
              |     Updated by Will Povell |
              |____________________________|


<?php echo $output; ?>
</pre>
</body>
</html>