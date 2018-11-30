import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class SwingApplication {
	public static void main(String[] args) {
		JFrame frame = new JFrame("Simple Swing application");
		Container contentPanel = frame.getContentPane();
		contentPanel.setLayout(new GridLayout(2, 1));
		JButton button = new JButton("Click me");
		final JLabel label = new JLabel();
		contentPanel.add(button);
		contentPanel.add(label);
		button.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				String information = JOptionPane.showInputDialog("Please input some string for label");
				label.setText(information);
			}
		});
		frame.setSize(800, 600);
		//frame.show();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
}