fileroot = 'qianshui';

units = 'km';
% %%
plotbdry3d qianshui.bty;

%%

bellhop3d( fileroot )

% %%
% figure
% plotbdry3d( [ fileroot '.bty' ] )
% hold on
% plotray3d( [ fileroot '.ray' ] )
% hold on
% % 从当前 env 文件读取接收器配置，并在射线图中标出全部接收器坐标
% envFile = [ fileroot '.env' ];
% if ~isfile( envFile )
% 	warning( 'env file not found: %s', envFile );
% else
% 	envText = fileread( envFile );
% 	envLines = strsplit( envText, newline );
% 
% 	sx = NaN;
% 	sy = NaN;
% 	sd = NaN;
% 	rdVals = [];
% 	nr = NaN;
% 	rRange = [ NaN NaN ];
% 	ntheta = NaN;
% 	bearingRange = [ NaN NaN ];
% 
% 	for i = 1 : numel( envLines )
% 		line = strtrim( envLines{ i } );
% 		if isempty( line )
% 			continue;
% 		end
% 
% 		if contains( line, '! x coordinate of source (km)' )
% 			vals = sscanf( line, '%f' );
% 			if ~isempty( vals )
% 				sx = vals( 1 );
% 			end
% 		elseif contains( line, '! y coordinate of source (km)' )
% 			vals = sscanf( line, '%f' );
% 			if ~isempty( vals )
% 				sy = vals( 1 );
% 			end
% 		elseif contains( line, '! SD(1:NSD)' )
% 			vals = sscanf( line, '%f' );
% 			if ~isempty( vals )
% 				sd = vals( 1 );
% 			end
% 		elseif contains( line, '! RD(1:NRD)' )
% 			vals = sscanf( line, '%f' );
% 			if ~isempty( vals )
% 				rdVals = vals(:).';
% 			end
% 		elseif contains( line, '! NR ' ) && ~contains( line, 'NRD' )
% 			vals = sscanf( line, '%f' );
% 			if ~isempty( vals )
% 				nr = round( vals( 1 ) );
% 			end
% 		elseif contains( line, '! R(1:NR ) (km)' )
% 			vals = sscanf( line, '%f' );
% 			if numel( vals ) >= 2
% 				rRange = vals( 1:2 ).';
% 			end
% 		elseif contains( line, '! Ntheta (number of bearings)' )
% 			vals = sscanf( line, '%f' );
% 			if ~isempty( vals )
% 				ntheta = round( vals( 1 ) );
% 			end
% 		elseif contains( line, '! bearing angles (degrees)' )
% 			vals = sscanf( line, '%f' );
% 			if numel( vals ) >= 2
% 				bearingRange = vals( 1:2 ).';
% 			end
% 		end
% 	end
% 
% 	missingRequired = any( isnan( [ sx sy nr ntheta rRange bearingRange ] ) ) || isempty( rdVals );
% 	if missingRequired
% 		warning( 'Failed to parse receiver settings from %s', envFile );
% 	else
% 		rSamples = linspace( rRange( 1 ), rRange( 2 ), nr );
% 		thetaSamples = linspace( bearingRange( 1 ), bearingRange( 2 ), ntheta );
% 		[ TH, RR ] = meshgrid( thetaSamples, rSamples );
% 
% 		% env中的x/y单位为km，这里统一转为m后再绘图
% 		sx_m = sx * 1000;
% 		sy_m = sy * 1000;
% 		xRecBase = ( sx + RR .* cosd( TH ) ) * 1000;
% 		yRecBase = ( sy + RR .* sind( TH ) ) * 1000;
% 
% 		% 标出声源位置
% 		if ~isnan( sd )
% 			plot3( sx_m, sy_m, sd, 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 10 );
% 			text( sx_m, sy_m, sd, sprintf( 'Source(%.1f m, %.1f m, %.1f m)', sx_m, sy_m, sd ), ...
% 				'FontSize', 7, 'Color', 'k', 'VerticalAlignment', 'bottom' );
% 		end
% 
% 		% 标出所有接收器并标注其坐标
% 		for k = 1 : numel( rdVals )
% 			zRec = rdVals( k ) * ones( size( xRecBase ) );
% 			scatter3( xRecBase(:), yRecBase(:), zRec(:), 18, 'r', 'filled' );
% 
% 			for j = 1 : numel( xRecBase )
% 				text( xRecBase( j ), yRecBase( j ), zRec( j ), ...
% 					sprintf( '(%.1f m, %.1f m, %.1f m)', xRecBase( j ), yRecBase( j ), zRec( j ) ), ...
% 					'FontSize', 6, 'Color', [ 0.85 0.1 0.1 ], 'HorizontalAlignment', 'left' );
% 			end
% 		end
% 
% 		title( sprintf( 'Ray Trace + Receivers (%d x %d x %d = %d)', ...
% 			nr, ntheta, numel( rdVals ), nr * ntheta * numel( rdVals ) ) );
% 	end
% end
% 
% %%
% figure
% % plotshdpol 参数：文件名，[源x坐标序列(km)],[源y坐标序列(km)], 深度(m)
% plotshdpol( [ fileroot '.shd' ], 17, 10, 200 )
% caxisrev( [ 60 100 ] )