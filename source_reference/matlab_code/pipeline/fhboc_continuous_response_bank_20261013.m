function bank=fhboc_continuous_response_bank_20261013(frame,cfg,options)
model=frame.operator_model;step=options.template_tau_step_samples;
grid=(floor(min(model.tau_axis))-3):step:(ceil(max(model.tau_axis))+3);
values=complex(zeros(size(model.operator,1),numel(grid)));
for first=1:16:numel(grid)
 ids=first:min(first+15,numel(grid));
 wave=fhboc_continuous_waveform_batch_20261013(frame.reference,cfg,model.start_s, ...
  grid(ids),repmat(frame.center_fd_hz,1,numel(ids)),zeros(1,numel(ids)));
 values(:,ids)=model.operator*wave;
end
bank=struct('tau_grid',grid,'response',values,'center_fd_hz',frame.center_fd_hz, ...
 'segment_times_s',((0:cfg.acq.P_segments-1)'+.5)*floor(frame.reference.frame_samples/cfg.acq.P_segments)/cfg.signal.fs, ...
 'template_tau_step_samples',step,'approximation','continuous reference anchors; segment-centre residual phase/delay');
end
