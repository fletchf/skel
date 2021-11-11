names = {'Angle','BendedLine','CShape','DoubleBendedLine','GShape',...
         'heee','JShape','JShape_2','Khamesh','Leaf_1',...
         'Leaf_2','Line','LShape','NShape','PShape',...
         'RShape','Saeghe','Sharpc','Sine','Snake',...
         'Spoon','Sshape','Trapezoid','Worm','WShape','Zshape'};

     
for n = 1:length(names)
    load(['DataSet/' names{n}],'demos','dt')

    demos_disc = cell(size(demos));
    t_final = zeros(size(demos));

    for i = 1:length(demos)

        t_final(i) = demos{i}.t(end);

    end


    dt_disc = 0.05;
    max_timesteps = floor(min(t_final) / dt_disc) + 1;

    for i = 1:length(demos)

        tvec = 0:dt_disc:t_final(i);

        pos_new = spline(demos{i}.t, demos{i}.pos, tvec);
        vel_new = spline(demos{i}.t, demos{i}.vel, tvec);
        acc_new = spline(demos{i}.t, demos{i}.acc, tvec);

        
        demos_disc{i}.pos = pos_new(:, 1:end);
        demos_disc{i}.vel = vel_new(:, 1:end);
        demos_disc{i}.acc = acc_new(:, 1:end);
        
        demos_disc{i}.pos(:, end) = [0; 0];
        demos_disc{i}.vel(:, end) = [0; 0];
        demos_disc{i}.acc(:, end) = [0; 0];
        
    end

    save(['DataDiscrete/' names{n}], 'demos', 'dt');

end
